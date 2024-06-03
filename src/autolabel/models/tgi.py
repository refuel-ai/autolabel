import asyncio
import json
import numpy as np
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

logger = logging.getLogger(__name__)


class TGILLM(BaseModel):
    DEFAULT_TOKENIZATION_MODEL = {
        "pretrained_model_name_or_path": "NousResearch/Llama-2-13b-chat-hf",
        "revision": "d73f5fa9c4bc135502e04c27b39660747172d76b",
    }
    DEFAULT_CONTEXT_LENGTH = 3250
    DEFAULT_CONNECT_TIMEOUT = 10
    DEFAULT_READ_TIMEOUT = 120
    DEFAULT_PARAMS = {
        "max_new_tokens": 128,
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
                "Unable to import transformers. Please install transformers to use TGILLM"
            )

        self.model_name = config.model_name()
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        if self.config.confidence():
            self.model_params["decoder_input_details"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            **self.DEFAULT_TOKENIZATION_MODEL
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _label_with_retry(self, prompt: str) -> Tuple[requests.Response, float]:
        payload = {"inputs": prompt, "parameters": {**self.model_params}}
        start_time = time()
        response = requests.post(
            self.model_params["endpoint"],
            json=payload,
            timeout=(self.DEFAULT_CONNECT_TIMEOUT, self.DEFAULT_READ_TIMEOUT),
        )
        end_time = time()
        # raise Exception if status != 200
        if response.status_code != 200:
            logger.warning(
                f"Received status code {response.status_code} from TGI server. Response: {response.text}"
            )
            response.raise_for_status()
        return response, end_time - start_time

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _alabel_with_retry(self, prompt: str) -> Tuple[requests.Response, float]:
        payload = {"inputs": prompt, "parameters": {**self.model_params}}
        async with httpx.AsyncClient() as client:
            timeout = httpx.Timeout(
                self.DEFAULT_CONNECT_TIMEOUT, read=self.DEFAULT_READ_TIMEOUT
            )
            start_time = time()
            response = await client.post(
                self.model_params["endpoint"], json=payload, timeout=timeout
            )
            end_time = time()
            # raise Exception if status != 200
            if response.status_code != 200:
                logger.warning(
                    f"Received status code {response.status_code} from TGI server. Response: {response.text}"
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
                response = json.loads(response.text)[0]
                generations.append(
                    [
                        Generation(
                            text=response["generated_text"],
                            generation_info=(
                                {
                                    "logprobs": {
                                        "top_logprobs": self._process_confidence_request(
                                            response["details"]["tokens"]
                                        )
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
                logger.exception(
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
                response = json.loads(response.text)[0]
                generations.append(
                    [
                        Generation(
                            text=response["generated_text"],
                            generation_info=(
                                {
                                    "logprobs": {
                                        "top_logprobs": self._process_confidence_request(
                                            response["details"]["tokens"]
                                        )
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
            logger.exception(
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

    def _process_confidence_request(self, logprobs: List):
        resp = []
        for item in logprobs:
            if not item.get("special", False):
                logprob = item["logprob"]
                if logprob is None:
                    logger.warning(f"Logprob is None! {item} {logprobs}")
                    logprob = 0.0
                resp.append({item["text"]: np.exp(logprob)})
        return resp
