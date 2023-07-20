from functools import cached_property
from typing import List, Optional
import os

import cohere
from langchain.llms import Cohere
from langchain.schema import LLMResult
from langchain import PromptTemplate, LLMChain

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache
from autolabel.schema import RefuelLLMResult


class CohereLLM(BaseModel):
    # Default parameters for OpenAILLM
    DEFAULT_MODEL = "command"
    DEFAULT_MODEL_PARAMS = {
        "max_tokens": 512,
        "temperature": 0.0,
    }

    # Reference: https://cohere.com/pricing
    COST_PER_TOKEN = 15 / 1_000_000

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)
        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL

        # populate model params and initialize the LLM
        model_params = config.model_params()
        self.model_params = {
            **self.DEFAULT_MODEL_PARAMS,
            **model_params,
        }
        self.llm = Cohere(model=self.model_name, **self.model_params)
        self.co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        try:
            result = self.llm.generate(prompts)
            return RefuelLLMResult(
                generations=result.generations, errors=[None] * len(result.generations)
            )
        except Exception as e:
            return self._label_individually(prompts)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        num_prompt_toks = len(self.co.tokenize(prompt).tokens)
        if label:
            num_label_toks = len(self.co.tokenize(label).tokens)
        else:
            num_label_toks = self.model_params["max_tokens"]

        return self.COST_PER_TOKEN * (num_prompt_toks + num_label_toks)

    def returns_token_probs(self) -> bool:
        return False
