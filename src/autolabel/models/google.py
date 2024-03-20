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
    CHAT_ENGINE_MODELS = ["gemini-pro"]

    DEFAULT_MODEL = "gemini-pro"

    # Reference: https://ai.google.dev/pricing
    COST_PER_PROMPT_CHARACTER = {
        "gemini-pro": 0.000125 / 1000,
    }

    COST_PER_COMPLETION_CHARACTER = {
        "gemini-pro": 0.000375 / 1000,
    }

    @cached_property
    def _engine(self) -> str:
        if self.model_name is not None and self.model_name in self.CHAT_ENGINE_MODELS:
            return "chat"
        else:
            return "completion"

    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
    ) -> None:
        try:
            import tiktoken
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
                HarmBlockThreshold,
                HarmCategory,
            )
        except ImportError:
            raise ImportError(
                "tiktoken and langchain_google_genai. Please install it with the following command: pip install 'refuel-autolabel[google]'"
            )

        self.DEFAULT_PARAMS = {
            "temperature": 0.0,
            "topK": 3,
            "safety_settings": {
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        }

        super().__init__(config, cache)

        if os.getenv("GOOGLE_API_KEY") is None:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL
        # populate model params and initialize the LLM
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        if self._engine == "chat":
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name, **self.model_params
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name, **self.model_params
            )

        self.tiktoken = tiktoken

    async def _alabel(self, prompts: List[str]) -> RefuelLLMResult:
        try:
            start_time = time()
            if self._engine == "chat":
                prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
                result = self.llm.agenerate(prompts)
            else:
                result = self.llm.agenerate(prompts)
            generations = result.generations
            end_time = time()
            return RefuelLLMResult(
                generations=generations,
                errors=[None] * len(generations),
                latencies=[end_time - start_time] * len(generations),
            )
        except Exception as e:
            return await self._alabel_individually(prompts)

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        try:
            start_time = time()
            if self._engine == "chat":
                prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
                result = self.llm.generate(prompts)
            else:
                result = self.llm.generate(prompts)
            generations = result.generations
            end_time = time()
            return RefuelLLMResult(
                generations=generations,
                errors=[None] * len(generations),
                latencies=[end_time - start_time] * len(generations),
            )
        except Exception as e:
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
