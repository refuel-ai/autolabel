import asyncio
import logging
import os
import re
from time import time
from typing import Dict, List, Optional, Tuple

import httpx
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

logger = logging.getLogger(__name__)


class UnretryableError(Exception):
    """This is an error which is unretriable from autolabel."""


class MiniMaxLLM(BaseModel):
    DEFAULT_CONNECT_TIMEOUT = 10
    DEFAULT_READ_TIMEOUT = 120
    DEFAULT_MODEL = "MiniMax-M2.7"
    DEFAULT_PARAMS = {
        "max_tokens": 1000,
        "temperature": 0.1,
    }

    # Reference: https://platform.minimaxi.com/document/Price
    COST_PER_PROMPT_TOKEN = {
        "MiniMax-M2.7": (1.0 / 1_000_000),
        "MiniMax-M2.5": (1.0 / 1_000_000),
        "MiniMax-M2.5-highspeed": (0.5 / 1_000_000),
    }
    COST_PER_COMPLETION_TOKEN = {
        "MiniMax-M2.7": (4.0 / 1_000_000),
        "MiniMax-M2.5": (4.0 / 1_000_000),
        "MiniMax-M2.5-highspeed": (2.0 / 1_000_000),
    }

    # MiniMax models may include <think>...</think> tags in responses
    THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        super().__init__(config, cache, tokenizer)

        if os.getenv("MINIMAX_API_KEY") is None:
            raise ValueError("MINIMAX_API_KEY environment variable not set")

        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL
        self.prompts2tokens = {}
        # populate model params
        model_params = config.model_params()
        self.timeout = self.DEFAULT_READ_TIMEOUT
        if "request_timeout" in model_params.keys():
            self.timeout = model_params["request_timeout"]
            del model_params["request_timeout"]
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        # Clamp temperature to MiniMax's accepted range [0, 1]
        if "temperature" in self.model_params:
            self.model_params["temperature"] = max(
                0.0, min(1.0, self.model_params["temperature"])
            )
        self.url = "https://api.minimax.io/v1/chat/completions"

    def _strip_think_tags(self, text: str) -> str:
        """Strip <think>...</think> tags from model response."""
        return self.THINK_TAG_PATTERN.sub("", text).strip()

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(UnretryableError),
    )
    def _label_with_retry(self, prompt: str) -> Tuple[requests.Response, float]:
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **self.model_params,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + os.getenv("MINIMAX_API_KEY"),
        }
        start_time = time()
        response = requests.post(
            self.url, json=data, headers=headers, timeout=self.timeout,
        )
        end_time = time()
        # raise Exception if status != 200
        if response.status_code != 200:
            logger.warning(
                f"Received status code {response.status_code} from MiniMax API. Response: {response.text}",
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
    async def _alabel_with_retry(self, prompt: str) -> Tuple[httpx.Response, float]:
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **self.model_params,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + os.getenv("MINIMAX_API_KEY"),
        }
        async with httpx.AsyncClient() as client:
            timeout = httpx.Timeout(self.DEFAULT_CONNECT_TIMEOUT, read=self.timeout)
            start_time = time()
            response = await client.post(
                self.url, json=data, headers=headers, timeout=timeout,
            )
            end_time = time()
            # raise Exception if status != 200
            if response.status_code != 200:
                logger.warning(
                    f"Received status code {response.status_code} from MiniMax API. Response: {response.text}",
                )
                response.raise_for_status()
            return response, end_time - start_time

    def _label(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        generations = []
        errors = []
        latencies = []
        for prompt in prompts:
            try:
                response, latency = self._label_with_retry(prompt)
                response = response.json()
                text = response["choices"][0]["message"]["content"]
                text = self._strip_think_tags(text)
                generations.append(
                    [Generation(text=text)],
                )
                errors.append(None)
                latencies.append(latency)
            except Exception as e:
                logger.error(
                    f"Unable to generate prediction: {e}",
                )
                generations.append([Generation(text="")])
                errors.append(
                    LabelingError(
                        error_type=ErrorType.LLM_PROVIDER_ERROR, error_message=str(e),
                    ),
                )
                latencies.append(0)
        return RefuelLLMResult(
            generations=generations, errors=errors, latencies=latencies,
        )

    async def _alabel(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        generations = []
        errors = []
        latencies = []
        try:
            requests = [self._alabel_with_retry(prompt) for prompt in prompts]
            responses = await asyncio.gather(*requests)
            for response, latency in responses:
                response = response.json()
                text = response["choices"][0]["message"]["content"]
                text = self._strip_think_tags(text)
                generations.append(
                    [Generation(text=text)],
                )
                errors.append(None)
                latencies.append(latency)
        except Exception as e:
            logger.error(
                f"Unable to generate prediction: {e}",
            )
            generations.append([Generation(text="")])
            errors.append(
                LabelingError(
                    error_type=ErrorType.LLM_PROVIDER_ERROR, error_message=str(e),
                ),
            )
            latencies.append(0)
        return RefuelLLMResult(
            generations=generations, errors=errors, latencies=latencies,
        )

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        cost_per_prompt_token = self.COST_PER_PROMPT_TOKEN.get(
            self.model_name, self.COST_PER_PROMPT_TOKEN["MiniMax-M2.7"]
        )
        cost_per_completion_token = self.COST_PER_COMPLETION_TOKEN.get(
            self.model_name, self.COST_PER_COMPLETION_TOKEN["MiniMax-M2.7"]
        )
        num_prompt_tokens = self.get_num_tokens(prompt)
        num_completion_tokens = self.get_num_tokens(label) if label else 0
        return (
            cost_per_prompt_token * num_prompt_tokens
            + cost_per_completion_token * num_completion_tokens
        )

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        return len(prompt) // 4
