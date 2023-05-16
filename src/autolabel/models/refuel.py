from typing import List, Optional
import json
from langchain.schema import LLMResult, Generation
from loguru import logger

from autolabel.models import BaseModel
from autolabel.configs import ModelConfig
from autolabel.cache import BaseCache

import requests

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)


class RefuelLLM(BaseModel):
    def __init__(self, config: ModelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)
        # populate model name
        # This is unused today, but in the future could
        # be used to decide which refuel model is queried
        self.model_name = config.get_model_name()
        self.model_params = {}

        # initialize runtime
        self.BASE_API = "https://api.refuel.ai/llm"

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, "WARNING"),
    )
    def _label_with_retry(self, prompt: str) -> requests.Response:
        payload = json.dumps({"model_input": prompt, "task": "generate"}).encode(
            "utf-8"
        )
        response = requests.post(self.BASE_API, data=payload)
        # raise Exception if status != 200
        response.raise_for_status()
        return response

    def _label(self, prompts: List[str]) -> LLMResult:
        generations = []
        for prompt in prompts:
            try:
                response = self._label_with_retry(prompt)
                generations.append([Generation(text=response.text.strip('"'))])
            except Exception as e:
                # This signifies an error in generating the response using RefuelLLm
                logger.error(
                    f"Unable to generate prediction: {e}",
                )
                generations.append([Generation(text="")])
        return LLMResult(generations=generations)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0
