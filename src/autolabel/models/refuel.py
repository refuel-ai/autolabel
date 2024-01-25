import json
import os
import requests
import logging
from typing import List, Optional, Tuple
from time import time

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


class RefuelLLM(BaseModel):
    DEFAULT_CONTEXT_LENGTH = 3500
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
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is required to use RefuelLLM. Please install it with the following command: pip install tiktoken"
            )

        # populate model name
        # This is unused today, but in the future could
        # be used to decide which refuel model is queried
        self.model_name = config.model_name()
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}

        # initialize runtime
        self.BASE_API = f"https://llm.refuel.ai/models/{self.model_name}/generate"
        self.REFUEL_API_ENV = "REFUEL_API_KEY"
        if self.REFUEL_API_ENV in os.environ and os.environ[self.REFUEL_API_ENV]:
            self.REFUEL_API_KEY = os.environ[self.REFUEL_API_ENV]
        else:
            raise ValueError(
                f"Did not find {self.REFUEL_API_ENV}, please add an environment variable"
                f" `{self.REFUEL_API_ENV}` which contains it"
            )
        self.tiktoken = tiktoken

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(UnretryableError),
    )
    def _label_with_retry(self, prompt: str) -> Tuple[requests.Response, float]:
        payload = {
            "input": prompt,
            "params": {**self.model_params},
            "confidence": self.config.confidence(),
        }
        headers = {"refuel_api_key": self.REFUEL_API_KEY}
        start_time = time()
        response = requests.post(self.BASE_API, json=payload, headers=headers)
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
                            generation_info={
                                "logprobs": {"top_logprobs": response["logprobs"]}
                            }
                            if self.config.confidence()
                            else None,
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
        # TODO(dhruva): Replace with actual tokenizer once that is pushed to cache
        encoding = self.tiktoken.encoding_for_model("gpt2")
        return len(encoding.encode(prompt)) * 1.3
