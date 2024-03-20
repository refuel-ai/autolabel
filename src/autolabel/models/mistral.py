import os
from time import time
from typing import List, Optional

from langchain.schema import HumanMessage

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import RefuelLLMResult


class MistralLLM(BaseModel):
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
        # populate model params
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        # initialize LLM
        self.llm = ChatMistralAI(model=self.model_name, **self.model_params)

    async def _alabel(self, prompts: List[str]) -> RefuelLLMResult:
        prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
        try:
            start_time = time()
            result = await self.llm.agenerate(prompts)
            end_time = time()
            return RefuelLLMResult(
                generations=result.generations,
                errors=[None] * len(result.generations),
                latencies=[end_time - start_time] * len(result.generations),
            )
        except Exception as e:
            return await self._alabel_individually(prompts)

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
        try:
            start_time = time()
            result = self.llm.generate(prompts)
            end_time = time()
            return RefuelLLMResult(
                generations=result.generations,
                errors=[None] * len(result.generations),
                latencies=[end_time - start_time] * len(result.generations),
            )
        except Exception as e:
            return self._label_individually(prompts)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        num_prompt_toks = len(prompt) // 4  # average token length is 4
        if label:
            num_label_toks = len(label) // 4  # average token length is 4
        else:
            # get an upper bound
            num_label_toks = self.model_params["max_tokens"]

        cost_per_prompt_token = self.COST_PER_PROMPT_TOKEN[self.model_name]
        cost_per_completion_token = self.COST_PER_COMPLETION_TOKEN[self.model_name]
        return (num_prompt_toks * cost_per_prompt_token) + (
            num_label_toks * cost_per_completion_token
        )

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        return len(prompt) // 4  # average token length is 4
