import copy
from typing import Dict
from enum import Enum
from langchain.llms import (
    Anthropic,
    BaseLLM,
    Cohere,
    OpenAI
)

from refuel_oracle.config import Config

# All available LLM providers
class LLMProvider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    cohere = "cohere"

# Default parameters that we will use to initialize LLMs from a provider
PROVIDER_TO_DEFAULT_PARAMS = {
    LLMProvider.openai: {
        "max_tokens": 30,
        "temperature": 0.0,
        "model_kwargs": {"logprobs": 1}
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
}


class LLMFactory:

    @staticmethod
    def _resolve_params(default_params: Dict, provided_params: Dict) -> Dict:
        final_params = copy.deepcopy(default_params)
        if not provided_params or type(provided_params) != "dict":
            return final_params

        for key, val in provided_params:
            final_params[key] = val
        return final_params

    @staticmethod
    def build_llm(config: Config) -> BaseLLM:
        # TODO: llm_model might need to be rolled up into llm_params in the future
        llm_provider = config.get_provider()
        llm_model = config.get_model_name()
        llm_params = config.get("model_params", {})
        llm_cls = PROVIDER_TO_LLM[llm_provider]
        llm_params = LLMFactory._resolve_params(
            PROVIDER_TO_DEFAULT_PARAMS[llm_provider], llm_params
        )
        if llm_provider == LLMProvider.openai:
            return llm_cls(model_name=llm_model, **llm_params)
        else:
            return llm_cls(model=llm_model, **llm_params)

    
