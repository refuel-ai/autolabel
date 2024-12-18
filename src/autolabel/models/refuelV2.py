import asyncio
import json
import logging
import os
from time import time
from typing import Dict, List, Optional, Tuple
from modal import Function
from modal.functions import FunctionCall
from fastapi import HTTPException
import re

import requests
from langchain.schema import Generation
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from transformers import AutoTokenizer

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import ErrorType, LabelingError, RefuelLLMResult

UNRETRYABLE_ERROR_CODES = [400, 422]
logger = logging.getLogger(__name__)


class UnretryableError(Exception):

    """This is an error which is unretriable from autolabel."""


async def request(func: Function, args):
    call = func.spawn(args)
    call_id = call.object_id
    function_call = FunctionCall.from_id(call_id)

    try:
        return function_call.get(timeout=60 * 59)
    except TimeoutError as e:
        raise HTTPException(504, f"Request to model timed out: {e}")


class RefuelLLMV2(BaseModel):
    DEFAULT_TOKENIZATION_MODEL = {
        "pretrained_model_name_or_path": "NousResearch/Llama-2-13b-chat-hf",
        "revision": "d73f5fa9c4bc135502e04c27b39660747172d76b",
    }
    DEFAULT_CONTEXT_LENGTH = 8192
    DEFAULT_CONNECT_TIMEOUT = 10
    DEFAULT_READ_TIMEOUT = 120
    DEFAULT_PARAMS = {
        "max_tokens": 128,
        "temperature": 0,
        "top_p": 1.0,
    }

    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        super().__init__(config, cache, tokenizer)
        try:
            from transformers import AutoTokenizer
        except Exception:
            raise Exception(
                "Unable to import transformers. Please install transformers to use RefuelLLM",
            )

        # populate model name
        # This is unused today, but in the future could
        # be used to decide which refuel model is queried
        self.model_name = config.model_name()
        model_params = config.model_params()
        self.max_context_length = config.max_context_length(
            default=self.DEFAULT_CONTEXT_LENGTH,
        )
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        self.model_endpoint = config.model_endpoint()
        self.tokenizer = (
            tokenizer
            if tokenizer
            else AutoTokenizer.from_pretrained(**self.DEFAULT_TOKENIZATION_MODEL)
        )
        self.read_timeout = self.model_params.get(
            "request_timeout",
            self.DEFAULT_READ_TIMEOUT,
        )
        self.adapter_path = self.model_params.get("adapter_id", None)
        if "request_timeout" in self.model_params:
            del self.model_params["request_timeout"]

        # initialize runtime
        self.REFUEL_API_ENV = "REFUEL_API_KEY"
        if os.environ.get(self.REFUEL_API_ENV):
            self.REFUEL_API_KEY = os.environ[self.REFUEL_API_ENV]
        else:
            raise ValueError(
                f"Did not find {self.REFUEL_API_ENV}, please add an environment variable"
                f" `{self.REFUEL_API_ENV}` which contains it",
            )

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(UnretryableError),
    )
    def _label_with_retry(
        self,
        prompt: str,
        output_schema: Dict,
    ) -> Tuple[requests.Response, float]:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "parameters": {**self.model_params},
            "confidence": self.config.confidence(),
            "adapter_path": self.adapter_path,
            "response_format": output_schema,
        }
        headers = {"refuel_api_key": self.REFUEL_API_KEY}
        start_time = time()
        response = requests.post(
            self.model_endpoint(),
            json=payload,
            headers=headers,
            timeout=(self.DEFAULT_CONNECT_TIMEOUT, self.read_timeout),
        )
        end_time = time()
        # raise Exception if status != 200
        if response.status_code != 200:
            if response.status_code in UNRETRYABLE_ERROR_CODES:
                # This is a bad request, and we should not retry
                raise UnretryableError(
                    f"NonRetryable Error: Received status code {response.status_code} from Refuel API. Response: {response.text}",
                )

            logger.warning(
                f"Received status code {response.status_code} from Refuel API. Response: {response.text}",
            )
            response.raise_for_status()
        return response, end_time - start_time

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(UnretryableError),
    )
    async def _alabel_with_retry(
        self,
        prompt: str,
        output_schema: Dict,
    ) -> Tuple[requests.Response, float]:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "parameters": {**self.model_params},
            "confidence": self.config.confidence(),
            "adapter_path": self.adapter_path,
            "response_format": output_schema,
        }
        start_time = time()
        match = re.search(r"/models/([^/]+)/", self.model_endpoint)
        if not match:
            raise UnretryableError(
                f"Unable to find model id in endpoint: {self.model_endpoint}"
            )
        model_id = match.group(1)
        llm = Function.lookup(model_id, "Model.generate")
        response = await request(llm, payload)
        return response, time() - start_time

    def _label(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        generations = []
        errors = []
        latencies = []
        for prompt in prompts:
            try:
                response, latency = self._label_with_retry(
                    prompt,
                    output_schema=output_schema,
                )
                response = json.loads(response.json())
                generations.append(
                    [
                        Generation(
                            text=response["generated_text"],
                            generation_info=(
                                {
                                    "logprobs": {
                                        "top_logprobs": response["details"][
                                            "output_logprobs"
                                        ],
                                    },
                                }
                                if self.config.confidence()
                                else None
                            ),
                        ),
                    ],
                )
                errors.append(None)
                latencies.append(latency)
            except Exception as e:
                # This signifies an error in generating the response using RefuelLLm
                logger.exception(
                    f"Unable to generate prediction: {e}",
                )
                generations.append([Generation(text="")])
                errors.append(
                    LabelingError(
                        error_type=ErrorType.LLM_PROVIDER_ERROR,
                        error_message=str(e),
                    ),
                )
                latencies.append(0)
        return RefuelLLMResult(
            generations=generations,
            errors=errors,
            latencies=latencies,
        )

    async def _alabel(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        generations = []
        errors = []
        latencies = []
        try:
            requests = [
                self._alabel_with_retry(prompt, output_schema=output_schema)
                for prompt in prompts
            ]
            responses = await asyncio.gather(*requests)
            for response, latency in responses:
                response = json.loads(response)
                generations.append(
                    [
                        Generation(
                            text=response["generated_text"],
                            generation_info=(
                                {
                                    "logprobs": {
                                        "top_logprobs": response["details"][
                                            "output_logprobs"
                                        ],
                                    },
                                }
                                if self.config.confidence()
                                else None
                            ),
                        ),
                    ],
                )
                errors.append(None)
                latencies.append(latency)
        except Exception as e:
            # This signifies an error in generating the response using RefuelLLm
            logger.exception(
                f"Unable to generate prediction: {e}",
            )
            generations.append([Generation(text="")])
            errors.append(
                LabelingError(
                    error_type=ErrorType.LLM_PROVIDER_ERROR,
                    error_message=str(e),
                ),
            )
            latencies.append(0)
        return RefuelLLMResult(
            generations=generations,
            errors=errors,
            latencies=latencies,
        )

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0

    def returns_token_probs(self) -> bool:
        return True

    def get_num_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))
