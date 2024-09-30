from time import time
from typing import Dict, List, Optional

from langchain.schema import HumanMessage, Generation

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import ErrorType, RefuelLLMResult, LabelingError


class AnthropicLLM(BaseModel):
    DEFAULT_MODEL = "claude-3-haiku-20240307"
    DEFAULT_PARAMS = {
        "max_tokens_to_sample": 1000,
        "temperature": 0.0,
    }

    # Reference: https://www.anthropic.com/api#pricing
    COST_PER_PROMPT_TOKEN = {
        "claude-instant-1.2": (0.80 / 1_000_000),
        "claude-2.0": (8 / 1_000_000),
        "claude-2.1": (8 / 1_000_000),
        "claude-3-opus-20240229": (15 / 1_000_000),
        "claude-3-sonnet-20240229": (3 / 1_000_000),
        "claude-3-haiku-20240307": (0.25 / 1_000_000),
        "claude-3-5-sonnet-20240620": (3 / 1_000_000),
    }
    COST_PER_COMPLETION_TOKEN = {
        "claude-instant-1.2": (2.4 / 1_000_000),
        "claude-2.0": (24 / 1_000_000),
        "claude-2.1": (24 / 1_000_000),
        "claude-3-opus-20240229": (75 / 1_000_000),
        "claude-3-sonnet-20240229": (15 / 1_000_000),
        "claude-3-haiku-20240307": (1.25 / 1_000_000),
        "claude-3-5-sonnet-20240620": (15 / 1_000_000),
    }

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)

        try:
            from anthropic._tokenizers import sync_get_tokenizer
            from langchain_anthropic import ChatAnthropic
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

    async def _alabel(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        try:
            prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
            start_time = time()
            result = await self.llm.agenerate(prompts)
            end_time = time()
            return RefuelLLMResult(
                generations=result.generations,
                errors=[None] * len(result.generations),
                latencies=[end_time - start_time] * len(result.generations),
            )
        except Exception as e:
            return RefuelLLMResult(
                generations=[[Generation(text="")] for _ in prompts],
                errors=[
                    LabelingError(
                        error_type=ErrorType.LLM_PROVIDER_ERROR,
                        error_message=str(e),
                    )
                    for _ in prompts
                ],
                latencies=[0 for _ in prompts],
            )

    def _label(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        try:
            prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
            start_time = time()
            result = self.llm.generate(prompts)
            end_time = time()
            return RefuelLLMResult(
                generations=result.generations,
                errors=[None] * len(result.generations),
                latencies=[end_time - start_time] * len(result.generations),
            )
        except Exception as e:
            return RefuelLLMResult(
                generations=[[Generation(text="")] for _ in prompts],
                errors=[
                    LabelingError(
                        error_type=ErrorType.LLM_PROVIDER_ERROR,
                        error_message=str(e),
                    )
                    for _ in prompts
                ],
                latencies=[0 for _ in prompts],
            )

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
