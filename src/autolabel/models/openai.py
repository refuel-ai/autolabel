from functools import cached_property
from typing import List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import LLMResult, HumanMessage, Generation
import tiktoken

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache


class OpenAILLM(BaseModel):
    CHAT_ENGINE_MODELS = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0613",
    ]
    MODELS_WITH_TOKEN_PROBS = ["text-curie-001", "text-davinci-003"]

    # Default parameters for OpenAILLM
    DEFAULT_MODEL = "gpt-3.5-turbo"
    DEFAULT_PARAMS_COMPLETION_ENGINE = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "model_kwargs": {"logprobs": 1},
    }
    DEFAULT_PARAMS_CHAT_ENGINE = {
        "max_tokens": 1000,
        "temperature": 0.0,
    }

    # Reference: https://openai.com/pricing
    COST_PER_PROMPT_TOKEN = {
        "text-davinci-003": 0.02 / 1000,
        "text-curie-001": 0.002 / 1000,
        "gpt-3.5-turbo": 0.0015 / 1000,
        "gpt-3.5-turbo-0613": 0.0015 / 1000,
        "gpt-3.5-turbo-16k": 0.003 / 1000,
        "gpt-3.5-turbo-16k-0613": 0.003 / 1000,
        "gpt-4": 0.03 / 1000,
        "gpt-4-0613": 0.03 / 1000,
        "gpt-4-32k": 0.06 / 1000,
        "gpt-4-32k-0613": 0.06 / 1000,
    }
    COST_PER_COMPLETION_TOKEN = {
        "text-davinci-003": 0.02 / 1000,
        "text-curie-001": 0.002 / 1000,
        "gpt-3.5-turbo": 0.002 / 1000,
        "gpt-3.5-turbo-0613": 0.002 / 1000,
        "gpt-3.5-turbo-16k": 0.004 / 1000,
        "gpt-3.5-turbo-16k-0613": 0.004 / 1000,
        "gpt-4": 0.06 / 1000,
        "gpt-4-0613": 0.06 / 1000,
        "gpt-4-32k": 0.12 / 1000,
        "gpt-4-32k-0613": 0.12 / 1000,
    }

    @cached_property
    def _engine(self) -> str:
        if self.model_name is not None and self.model_name in self.CHAT_ENGINE_MODELS:
            return "chat"
        else:
            return "completion"

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)
        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL

        # populate model params and initialize the LLM
        model_params = config.model_params()
        if self._engine == "chat":
            self.model_params = {**self.DEFAULT_PARAMS_CHAT_ENGINE, **model_params}
            self.llm = ChatOpenAI(model_name=self.model_name, **self.model_params)
        else:
            self.model_params = {
                **self.DEFAULT_PARAMS_COMPLETION_ENGINE,
                **model_params,
            }
            self.llm = OpenAI(model_name=self.model_name, **self.model_params)

    def _label(self, prompts: List[str]) -> LLMResult:
        if self._engine == "chat":
            # Need to convert list[prompts] -> list[messages]
            # Currently the entire prompt is stuck into the "human message"
            # We might consider breaking this up into human vs system message in future
            prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
        try:
            return self.llm.generate(prompts)
        except Exception as e:
            print(f"Error generating from LLM: {e}, returning empty result")
            generations = [[Generation(text="")] for _ in prompts]
            return LLMResult(generations=generations)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        encoding = tiktoken.encoding_for_model(self.model_name)
        num_prompt_toks = len(encoding.encode(prompt))
        if label:
            num_label_toks = len(encoding.encode(label))
        else:
            # get an upper bound
            num_label_toks = self.model_params["max_tokens"]

        cost_per_prompt_token = self.COST_PER_PROMPT_TOKEN[self.model_name]
        cost_per_completion_token = self.COST_PER_COMPLETION_TOKEN[self.model_name]
        return (num_prompt_toks * cost_per_prompt_token) + (
            num_label_toks * cost_per_completion_token
        )

    def returns_token_probs(self) -> bool:
        return (
            self.model_name is not None
            and self.model_name in self.MODELS_WITH_TOKEN_PROBS
        )
