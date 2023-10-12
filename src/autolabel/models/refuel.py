import json
import os
import requests
import logging
from typing import List, Optional

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache
from autolabel.schema import LabelingError, ErrorType, RefuelLLMResult

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)
from langchain.schema import Generation

logger = logging.getLogger(__name__)


class RefuelLLM(BaseModel):
    DEFAULT_PARAMS = {
        "max_new_tokens": 128,
    }

    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
    ) -> None:
        super().__init__(config, cache)

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

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _label_with_retry(self, prompt: str) -> requests.Response:
        payload = {
            "input": prompt,
            "params": {**self.model_params},
            "confidence": self.config.confidence(),
        }
        headers = {"refuel_api_key": self.REFUEL_API_KEY}
        response = requests.post(self.BASE_API, json=payload, headers=headers)
        # raise Exception if status != 200
        response.raise_for_status()
        return response

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        generations = []
        errors = []
        for prompt in prompts:
            try:
                response = self._label_with_retry(prompt)
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
        return RefuelLLMResult(generations=generations, errors=errors)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0

    def returns_token_probs(self) -> bool:
        return True
