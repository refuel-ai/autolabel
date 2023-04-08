import copy
from typing import Dict, List
from enum import Enum
from langchain.chat_models import ChatOpenAI
from langchain.llms import (
    Anthropic,
    BaseLLM,
    Cohere,
    OpenAI
)
from langchain.schema import (
    HumanMessage,
    LLMResult
)

from refuel_oracle.config import Config

# All available LLM providers
class LLMProvider(str, Enum):
    openai = "openai"
    openai_chat = "openai_chat"
    anthropic = "anthropic"
    cohere = "cohere"


class LLMLabeler:
    def __init__(self, config: Config, base_llm: BaseLLM) -> None:
        self.config = config
        self.base_llm = base_llm
    
    def is_chat_model(self, llm_provider: LLMProvider) -> bool:
        # Add more models here that are in `langchain.chat_models.*`
        return llm_provider == LLMProvider.openai_chat

    def generate(self, prompts: List) -> LLMResult:
        llm_provider = self.config.get_provider()
        if self.is_chat_model(llm_provider):
            # Need to convert list[prompts] -> list[messages]
            # Currently the entire prompt is stuck into the "human message"
            # We might consider breaking this up into human vs system message in future
            prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
        return self.base_llm.generate(prompts)

class LLMFactory:

    # Default parameters that we will use to initialize LLMs from a provider
    PROVIDER_TO_DEFAULT_PARAMS = {
        LLMProvider.openai: {
            "max_tokens": 30,
            "temperature": 0.0,
            "model_kwargs": {"logprobs": 1}
        },
        LLMProvider.openai_chat: {
            "max_tokens": 30,
            "temperature": 0.0,
        },
        LLMProvider.cohere: {
            # TODO
        },
        LLMProvider.anthropic: {
            # TODO
        }
    }

    # Provider mapping to the langchain LLM wrapper
    PROVIDER_TO_LLM = {
        LLMProvider.anthropic: Anthropic,
        LLMProvider.cohere: Cohere,
        LLMProvider.openai: OpenAI,
        LLMProvider.openai_chat: ChatOpenAI
    }

    @staticmethod
    def _resolve_params(default_params: Dict, provided_params: Dict) -> Dict:
        final_params = copy.deepcopy(default_params)
        if not provided_params or type(provided_params) != "dict":
            return final_params

        for key, val in provided_params:
            final_params[key] = val
        return final_params

    @staticmethod
    def from_config(config: Config) -> LLMLabeler:
        # TODO: llm_model might need to be rolled up into llm_params in the future
        llm_provider = config.get_provider()
        llm_model = config.get_model_name()
        llm_params = config.get("model_params", {})
        llm_cls = LLMFactory.PROVIDER_TO_LLM[llm_provider]
        llm_params = LLMFactory._resolve_params(
            LLMFactory.PROVIDER_TO_DEFAULT_PARAMS[llm_provider], llm_params
        )
        if llm_provider in [LLMProvider.openai, LLMProvider.openai_chat]:
            base_llm = llm_cls(model_name=llm_model, **llm_params)
        else:
            base_llm = llm_cls(model=llm_model, **llm_params)
        return LLMLabeler(config, base_llm)
    
