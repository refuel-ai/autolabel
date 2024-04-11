import os
import logging
from functools import cached_property
from time import time
from typing import List, Optional

from langchain.schema import Generation, HumanMessage, LLMResult
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import ErrorType, LabelingError, RefuelLLMResult

logger = logging.getLogger(__name__)


class GoogleLLM(BaseModel):
    SEP_REPLACEMENT_TOKEN = "@@"
    CHAT_ENGINE_MODELS = ["gemini-pro", "gemini-1.5-pro-preview-0409"]

    DEFAULT_MODEL = "gemini-pro"

    # Reference: https://ai.google.dev/pricing
    COST_PER_PROMPT_CHARACTER = {
        "gemini-pro": 0.5 / 1_000_000,
        "gemini-1.5-pro-preview-0409": 7 / 1_000_000,
    }

    COST_PER_COMPLETION_CHARACTER = {
        "gemini-pro": 1.5 / 1_000_000,
        "gemini-1.5-pro-preview-0409": 21 / 1_000_000,
    }

    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
    ) -> None:
        try:
            import tiktoken
            from langchain_google_vertexai import (
                VertexAI,
                HarmBlockThreshold,
                HarmCategory,
            )
        except ImportError:
            raise ImportError(
                "tiktoken and langchain_google_vertexai. Please install it with the following command: pip install 'refuel-autolabel[google]'"
            )

        self.DEFAULT_PARAMS = {
            "temperature": 0.0,
            "topK": 3,
            "safety_settings": {
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
        }

        super().__init__(config, cache)

        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
            )

        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL
        # populate model params and initialize the LLM
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        self.llm = VertexAI(model_name=self.model_name, **self.model_params)
        self.tiktoken = tiktoken

    async def _alabel(self, prompts: List[str]) -> RefuelLLMResult:
        try:
            start_time = time()
            result = await self.llm.agenerate(prompts)
            generations = result.generations
            end_time = time()
            return RefuelLLMResult(
                generations=generations,
                errors=[None] * len(generations),
                latencies=[end_time - start_time] * len(generations),
            )
        except Exception as e:
            logger.error(f"Error from Google LLM: {e}")
            return await self._alabel_individually(prompts)

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        try:
            start_time = time()
            result = self.llm.generate(prompts)
            generations = result.generations
            end_time = time()
            return RefuelLLMResult(
                generations=generations,
                errors=[None] * len(generations),
                latencies=[end_time - start_time] * len(generations),
            )
        except Exception as e:
            logger.error(f"Error from Google LLM: {e}")
            return self._label_individually(prompts)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        if self.model_name is None:
            return 0.0
        cost_per_prompt_char = self.COST_PER_PROMPT_CHARACTER[self.model_name]
        cost_per_completion_char = self.COST_PER_COMPLETION_CHARACTER[self.model_name]
        return cost_per_prompt_char * len(prompt) + cost_per_completion_char * (
            len(label) if label else 0.0
        )

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        return super().get_num_tokens(prompt)
