from typing import List, Optional

from anthropic import tokenizer
from langchain.llms import Anthropic
from langchain.schema import LLMResult, Generation

from autolabel.models import BaseModel
from autolabel.configs import ModelConfig


class AnthropicLLM(BaseModel):
    MODELS = ["claude-v1", "claude-instant-v1"]
    DEFAULT_MODEL = "claude-v1"
    DEFAULT_PARAMS = {
        "max_tokens_to_sample": 1000,
        "temperature": 0.0,
    }

    # Reference: https://cdn2.assets-servd.host/anthropic-website/production/images/apr-pricing-tokens.pdf
    COST_PER_PROMPT_TOKEN = {
        # $11.02 per million tokens
        "claude-v1": (11.02 / 1000000),
        # $1.63 per million tokens
        "claude-instant-v1": (1.63 / 1000000),
    }
    COST_PER_COMPLETION_TOKEN = {
        # $32.68 per million tokens
        "claude-v1": (32.68 / 1000000),
        # $5.51 per million tokens
        "claude-instant-v1": (5.51 / 1000000),
    }

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        # populate model name
        self.model_name = config.get_model_name() or self.DEFAULT_MODEL
        # populate model params
        model_params = config.get_model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        # initialize LLM
        self.llm = Anthropic(model=self.model_name, **self.model_params)

    def label(self, prompts: List[str]) -> LLMResult:
        try:
            response = self.llm.generate(prompts)
            return response
        except Exception as e:
            print(f"Error generating from LLM: {e}, returning empty result")
            generations = [[Generation(text="")] for _ in prompts]
            return LLMResult(generations=generations)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        num_prompt_toks = tokenizer.count_tokens(prompt)
        if label:
            num_label_toks = tokenizer.count_tokens(label)
        else:
            # get an upper bound
            num_label_toks = self.model_params["max_tokens_to_sample"]

        cost_per_prompt_token = self.COST_PER_PROMPT_TOKEN[self.model_name]
        cost_per_completion_token = self.COST_PER_COMPLETION_TOKEN[self.model_name]
        return (num_prompt_toks * cost_per_prompt_token) + (
            num_label_toks * cost_per_completion_token
        )
