from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List

import openai
from openai.error import AuthenticationError as OpenAIAuthenticationError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


class LLMProvider(str, Enum):
    openai = "openai"


class LLMResults(BaseModel):
    completions: List[Dict]
    usage_cost: float
    usage_num_tokens: int


class LLM(ABC):
    def __init__(
        self, provider: LLMProvider, model_name: str, params: Dict = None
    ) -> None:
        """
        Create a new instance of an LLM client.

        Args:
            provider (LLMProvider): Provider for the LLM (currently, only openai supported)
            model_name (str): Model name from the provider. Model names supported:
            openai: ["text-davinci-003", "text-curie-001"]

        """
        self.model_name = model_name
        self.provider = provider
        self.params = self._resolve_params(params)

    @abstractmethod
    def _default_params(self) -> Dict:
        return "Not Implemented"

    def _resolve_params(self, provided_params: Dict) -> Dict:
        final_params = self._default_params()
        if not provided_params or type(provided_params) != "dict":
            return final_params

        for key, val in provided_params:
            final_params[key] = val
        return final_params

    # TODO: providing this as an async API
    @abstractmethod
    def generate(self, prompts: List[str]) -> LLMResults:
        """
        This function generate outputs from this LLM for a list of prompts.

        Returns:
            List[LLMResult]: List of results from the LLM.
        """
        return "Not Implemented"


class OpenAI(LLM):
    SUPPORTED_MODELS = set(["text-davinci-003", "text-curie-001"])
    PRICING_PER_TOKEN = {
        "text-davinci-003": 0.02 / 1000,
        "text-curie-001": 0.002 / 1000,
    }

    def __init__(
        self, model_name: str = "text-davinci-003", params: Dict = None
    ) -> None:
        if model_name not in OpenAI.SUPPORTED_MODELS:
            error_message = f"Model name: {model_name} not supported by provider: {LLMProvider.openai}"
            raise ValueError(error_message)
        self._validate_environment()
        super().__init__(LLMProvider.openai, model_name, params)

    def _validate_environment(self):
        try:
            import openai
        except ImportError:
            raise ValueError(
                "Could not import `openai` python package. "
                "Please install it with `pip install openai`."
            )

    def _default_params(self) -> Dict:
        return {"max_tokens": 30, "temperature": 0.0, "logprobs": 5}

    def _cost(self, num_tokens) -> float:
        return num_tokens * OpenAI.PRICING_PER_TOKEN[self.model_name]

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(OpenAIAuthenticationError),
    )
    def generate(self, prompts: List[str]) -> LLMResults:
        response = openai.Completion.create(
            model=self.model_name, prompt=prompts, **self.params
        )
        num_tokens = response.usage.total_tokens
        llm_results = []
        for item in response.choices:
            llm_results.append(item.to_dict_recursive())

        return LLMResults(
            completions=llm_results,
            usage_num_tokens=num_tokens,
            usage_cost=self._cost(num_tokens),
        )
