from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List

import openai
from loguru import logger
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


class LLMResult(BaseModel):
    completion: Dict


class LLMResults(BaseModel):
    completions: List[LLMResult]
    usage: Dict


class LLM(ABC):
    def __init__(
        self, model_name: str, provider: LLMProvider, params: Dict = None
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self.params = self._resolve_params(params)

    @abstractmethod
    def _default_params(self) -> Dict:
        return "Not Implemented"

    def _resolve_params(self, provided_params) -> Dict:
        final_params = self._default_params()
        if not provided_params or type(provided_params) != "dict":
            return final_params

        for key, val in provided_params:
            final_params[key] = val
        return final_params

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[LLMResult]:
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
        super().__init__(model_name, LLMProvider.openai, params)

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
    def generate(self, prompts: List[str]) -> List[LLMResults]:
        response = openai.Completion.create(
            model=self.model_name, prompt=prompts, **self.params
        )
        num_tokens = response.usage.total_tokens
        logger.info(num_tokens)
        logger.info(len(response.choices))
        llm_results = []
        for item in response.choices:
            llm_results.append(LLMResult(completion=item.to_dict_recursive()))

        logger.info(response)

        return LLMResults(
            completions=llm_results,
            usage={"tokens": num_tokens, "cost": self._cost(num_tokens)},
        )
