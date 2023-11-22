from typing import List, Optional
from time import time

from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.cache import BaseCache
from autolabel.schema import RefuelLLMResult
from langchain.schema import HumanMessage


class AnthropicLLM(BaseModel):
    DEFAULT_MODEL = "claude-instant-v1"
    DEFAULT_PARAMS = {
        "max_tokens_to_sample": 1000,
        "temperature": 0.0,
    }

    # Reference: https://cdn2.assets-servd.host/anthropic-website/production/images/apr-pricing-tokens.pdf
    COST_PER_PROMPT_TOKEN = {
        # $11.02 per million tokens
        "claude-v1": (11.02 / 1000000),
        "claude-instant-v1": (1.63 / 1000000),
    }
    COST_PER_COMPLETION_TOKEN = {
        # $32.68 per million tokens
        "claude-v1": (32.68 / 1000000),
        "claude-instant-v1": (5.51 / 1000000),
    }

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)

        try:
            from langchain.chat_models import ChatAnthropic
            from anthropic._tokenizers import sync_get_tokenizer
        except ImportError:
            raise ImportError(
                "anthropic is required to use the anthropic LLM. Please install it with the following command: pip install 'refuel-autolabel[anthropic]'"
            )

        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL
        # populate model params
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        # initialize LLM
        self.llm = ChatAnthropic(model=self.model_name, **self.model_params)

        self.tokenizer = sync_get_tokenizer()

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
        num_prompt_toks = len(self.tokenizer.encode(prompt).ids)
        if label:
            num_label_toks = len(self.tokenizer.encode(label).ids)
        else:
            # get an upper bound
            num_label_toks = self.model_params["max_tokens_to_sample"]

        cost_per_prompt_token = self.COST_PER_PROMPT_TOKEN[self.model_name]
        cost_per_completion_token = self.COST_PER_COMPLETION_TOKEN[self.model_name]
        return (num_prompt_toks * cost_per_prompt_token) + (
            num_label_toks * cost_per_completion_token
        )

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt).ids)
