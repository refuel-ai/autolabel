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


class VLLMModel(BaseModel):
    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
    ) -> None:
        super().__init__(config, cache)
        self.BASE_API = f"http://localhost:8000/v1/completions"
        self.model_name = config.model_name()

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(UnretryableError),
    )
    def _label_with_retry(self, prompt: str) -> Tuple[requests.Response, float]:
        payload = {
            "model": self.config.model_name(),
            "prompt": prompt,
            **self.model_params,
        }
        response = requests.post(self.BASE_API, json=payload)
        response.raise_for_status()
        return response

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        generations = []
        errors = []
        latencies = []
        for prompt in prompts:
            latencies.append(0)
            try:
                response = self._label_with_retry(prompt)
                response = response.json()
                generations.append(
                    [
                        Generation(
                            text=response["choices"][0]["text"],
                        )
                    ]
                )
                errors.append(None)
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
        return RefuelLLMResult(
            generations=generations, errors=errors, latencies=latencies
        )

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        return len(prompt)
