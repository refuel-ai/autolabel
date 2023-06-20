import json
import os
import requests
from langchain.schema import LLMResult, Generation
import logging
from typing import List, Optional


from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache


from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class RefuelLLM(BaseModel):
    DEFAULT_PARAMS = {
        "max_new_tokens": 128,
        "temperature": 0.0,
    }

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)
        # populate model name
        # This is unused today, but in the future could
        # be used to decide which refuel model is queried
        self.model_name = config.model_name()
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}

        # initialize runtime
        self.BASE_API = "https://refuel-llm.refuel.ai/"
        self.SEP_REPLACEMENT_TOKEN = "@@"
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
            "data": {"model_input": prompt, "model_params": {**self.model_params}},
            "task": "generate",
        }
        headers = {"refuel_api_key": self.REFUEL_API_KEY}
        response = requests.post(self.BASE_API, json=payload, headers=headers)
        # raise Exception if status != 200
        response.raise_for_status()
        return response

    def _label(self, prompts: List[str]) -> LLMResult:
        generations = []
        for prompt in prompts:
            try:
                if self.SEP_REPLACEMENT_TOKEN in prompt:
                    logger.warning(
                        f"""Current prompt contains {self.SEP_REPLACEMENT_TOKEN} 
                            which is currently used as a separator token by refuel
                            llm. It is highly recommended to avoid having any
                            occurences of this substring in the prompt.
                        """
                    )
                separated_prompt = prompt.replace("\n", self.SEP_REPLACEMENT_TOKEN)
                response = self._label_with_retry(separated_prompt)
                response = json.loads(response.json()["body"]).replace(
                    self.SEP_REPLACEMENT_TOKEN, "\n"
                )
                generations.append([Generation(text=response)])
            except Exception as e:
                # This signifies an error in generating the response using RefuelLLm
                logger.error(
                    f"Unable to generate prediction: {e}",
                )
                generations.append([Generation(text="")])
        return LLMResult(generations=generations)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0

    def returns_token_probs(self) -> bool:
        return False
