import asyncio
import json
import os
import requests
import logging
from typing import List, Optional, Tuple
from time import time
import httpx

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache
from autolabel.schema import LabelingError, ErrorType, RefuelLLMResult

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_not_exception_type,
)
from langchain.schema import Generation

UNRETRYABLE_ERROR_CODES = [400, 422]
logger = logging.getLogger(__name__)


class UnretryableError(Exception):
    """This is an error which is unretriable from autolabel."""


class RefuelLLMV2(BaseModel):
    DEFAULT_TOKENIZATION_MODEL = {
        "pretrained_model_name_or_path": "NousResearch/Llama-2-13b-chat-hf",
        "revision": "d73f5fa9c4bc135502e04c27b39660747172d76b",
    }
    DEFAULT_CONTEXT_LENGTH = 3250
    DEFAULT_CONNECT_TIMEOUT = 10
    DEFAULT_READ_TIMEOUT = 120
    DEFAULT_PARAMS = {
        "max_tokens": 128,
        "temperature": 0.05,
        "top_p": 1.0,
    }

    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
    ) -> None:
        super().__init__(config, cache)
        try:
            from transformers import AutoTokenizer
        except Exception as e:
            raise Exception(
                "Unable to import transformers. Please install transformers to use RefuelLLM"
            )

        # populate model name
        # This is unused today, but in the future could
        # be used to decide which refuel model is queried
        self.model_name = config.model_name()
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        self.model_endpoint = config.model_endpoint()
        self.tokenizer = AutoTokenizer.from_pretrained(
            **self.DEFAULT_TOKENIZATION_MODEL
        )
        self.read_timeout = self.model_params.get(
            "request_timeout", self.DEFAULT_READ_TIMEOUT
        )
        self.adapter_path = self.model_params.get("adapter_id", None)
        del self.model_params["request_timeout"]

        # initialize runtime
        self.REFUEL_API_ENV = "REFUEL_API_KEY"
        if self.REFUEL_API_ENV in os.environ and os.environ[self.REFUEL_API_ENV]:
            self.REFUEL_API_KEY = os.environ[self.REFUEL_API_ENV]
        else:
            raise ValueError(
                f"Did not find {self.REFUEL_API_ENV}, please add an environment variable"
                f" `{self.REFUEL_API_ENV}` which contains it"
            )

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(UnretryableError),
    )
    def _label_with_retry(self, prompt: str) -> Tuple[requests.Response, float]:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "parameters": {**self.model_params},
            "confidence": self.config.confidence(),
            "adapter_path": self.adapter_path,
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
                    f"NonRetryable Error: Received status code {response.status_code} from Refuel API. Response: {response.text}"
                )

            logger.warning(
                f"Received status code {response.status_code} from Refuel API. Response: {response.text}"
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
    async def _alabel_with_retry(self, prompt: str) -> Tuple[requests.Response, float]:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "parameters": {**self.model_params},
            "confidence": self.config.confidence(),
            "adapter_path": self.adapter_path,
        }
        headers = {"refuel_api_key": self.REFUEL_API_KEY}
        async with httpx.AsyncClient() as client:
            timeout = httpx.Timeout(
                self.DEFAULT_CONNECT_TIMEOUT, read=self.read_timeout
            )
            start_time = time()
            response = await client.post(
                self.model_endpoint, json=payload, headers=headers, timeout=timeout
            )
            end_time = time()
            # raise Exception if status != 200
            if response.status_code != 200:
                if response.status_code in UNRETRYABLE_ERROR_CODES:
                    # This is a bad request, and we should not retry
                    raise UnretryableError(
                        f"NonRetryable Error: Received status code {response.status_code} from Refuel API. Response: {response.text}"
                    )

                logger.warning(
                    f"Received status code {response.status_code} from Refuel API. Response: {response.text}"
                )
                response.raise_for_status()
            return response, end_time - start_time

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        generations = []
        errors = []
        latencies = []
        for prompt in prompts:
            try:
                response, latency = self._label_with_retry(prompt)
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
                                        ]
                                    }
                                }
                                if self.config.confidence()
                                else None
                            ),
                        )
                    ]
                )
                errors.append(None)
                latencies.append(latency)
            except Exception as e:
                # This signifies an error in generating the response using RefuelLLm
                logger.error(
                    f"Unable to generate prediction: {e}",
                )
                generations.append([Generation(text="")])
                errors.append(
                    LabelingError(
                        error_type=ErrorType.LLM_PROVIDER_ERROR, error_message=str(e)
                    )
                )
                latencies.append(0)
        return RefuelLLMResult(
            generations=generations, errors=errors, latencies=latencies
        )

    async def _alabel(self, prompts: List[str]) -> RefuelLLMResult:
        generations = []
        errors = []
        latencies = []
        try:
            requests = [self._alabel_with_retry(prompt) for prompt in prompts]
            responses = await asyncio.gather(*requests)
            for response, latency in responses:
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
                                        ]
                                    }
                                }
                                if self.config.confidence()
                                else None
                            ),
                        )
                    ]
                )
                errors.append(None)
                latencies.append(latency)
        except Exception as e:
            # This signifies an error in generating the response using RefuelLLm
            logger.error(
                f"Unable to generate prediction: {e}",
            )
            generations.append([Generation(text="")])
            errors.append(
                LabelingError(
                    error_type=ErrorType.LLM_PROVIDER_ERROR, error_message=str(e)
                )
            )
            latencies.append(0)
        return RefuelLLMResult(
            generations=generations, errors=errors, latencies=latencies
        )

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0

    def returns_token_probs(self) -> bool:
        return True

    def get_num_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))
