import asyncio
import os
import requests
from time import time
from typing import List, Optional, Tuple
import httpx

from langchain.schema import HumanMessage

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import LabelingError, ErrorType, RefuelLLMResult
import json
import logging
from transformers import AutoTokenizer

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_not_exception_type,
)

from langchain.schema import Generation

logger = logging.getLogger(__name__)


class UnretryableError(Exception):
    """This is an error which is unretriable from autolabel."""


class MistralLLM(BaseModel):
    DEFAULT_TOKENIZATION_MODEL = "NousResearch/Llama-2-13b-chat-hf"
    DEFAULT_CONTEXT_LENGTH = 3250
    DEFAULT_CONNECT_TIMEOUT = 10
    DEFAULT_READ_TIMEOUT = 120
    DEFAULT_MODEL = "mistral-small-latest"
    DEFAULT_PARAMS = {
        "max_tokens": 1000,
        "temperature": 0.0,
    }

    # Reference: https://docs.mistral.ai/platform/pricing/
    COST_PER_PROMPT_TOKEN = {
        "open-mistral-7b": (0.25 / 1_000_000),
        "open-mixtral-8x7b": (0.7 / 1_000_000),
        "mistral-small-latest": (2 / 1_000_000),
        "mistral-medium-latest": (2.7 / 1_000_000),
        "mistral-large-latest": (8 / 1_000_000),
    }
    COST_PER_COMPLETION_TOKEN = {
        "open-mistral-7b": (0.25 / 1_000_000),
        "open-mixtral-8x7b": (0.7 / 1_000_000),
        "mistral-small-latest": (6 / 1_000_000),
        "mistral-medium-latest": (8.1 / 1_000_000),
        "mistral-large-latest": (24 / 1_000_000),
    }

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)

        try:
            # get the tokenizer
            from langchain_mistralai import ChatMistralAI
        except ImportError:
            raise ImportError(
                "mistralai and langchain_mistralai is required to use the anthropic LLM. Please install it with the following command: pip install 'refuel-autolabel[mistral]'"
            )

        if os.getenv("MISTRAL_API_KEY") is None:
            raise ValueError("MISTRAL_API_KEY environment variable not set")

        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.prompts2tokens = {}
        # populate model params
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        self.url = "https://api.mistral.ai/v1/chat/completions"

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
            "Authorization": "Bearer " + os.getenv("MISTRAL_API_KEY"),
        }
        start_time = time()
        response = requests.post(self.url, json=data, headers=headers)
        end_time = time()
        # raise Exception if status != 200
        if response.status_code != 200:
            logger.warning(
                f"Received status code {response.status_code} from Mistral API. Response: {response.text}"
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
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **self.model_params,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + os.getenv("MISTRAL_API_KEY"),
        }
        async with httpx.AsyncClient() as client:
            timeout = httpx.Timeout(
                self.DEFAULT_CONNECT_TIMEOUT, read=self.DEFAULT_READ_TIMEOUT
            )
            start_time = time()
            response = await client.post(
                self.url, json=data, headers=headers, timeout=timeout
            )
            end_time = time()
            # raise Exception if status != 200
            if response.status_code != 200:
                logger.warning(
                    f"Received status code {response.status_code} from Mistral API. Response: {response.text}"
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
                response = response.json()
                generations.append(
                    [
                        Generation(
                            text=response["choices"][0]["message"]["content"],
                            generation_info=(
                                {
                                    "logprobs": {
                                        "top_logprobs": response["choices"][0][
                                            "logprobs"
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
                response = response.json()
                generations.append(
                    [
                        Generation(
                            text=response["choices"][0]["message"]["content"],
                            generation_info=(
                                {
                                    "logprobs": {
                                        "top_logprobs": response["choices"][0][
                                            "logprobs"
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
            print(e)
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
        cost_per_prompt_char = self.COST_PER_PROMPT_TOKEN[self.model_name]
        cost_per_completion_char = self.COST_PER_COMPLETION_TOKEN[self.model_name]
        return cost_per_prompt_char * len(prompt) + cost_per_completion_char * (
            len(label) if label else 0.0
        )

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))
