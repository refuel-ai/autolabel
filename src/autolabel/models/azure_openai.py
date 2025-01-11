
import json
import logging
import os
from functools import partial
from time import time
from typing import Dict, List, Optional

from langchain.schema import Generation
from transformers import AutoTokenizer

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import ErrorType, LabelingError, RefuelLLMResult

logger = logging.getLogger(__name__)


class AzureOpenAILLM(BaseModel):
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_PARAMS = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "timeout": 30,
        "logprobs": True,
        "stream": False
    }

    # Reference: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
    COST_PER_PROMPT_TOKEN = {
        "gpt-35-turbo": 1 / 1_000_000,
        "gpt-4-turbo-2024-04-09": 10 / 1_000_000,
        "gpt-4o": 2.50 / 1_000_000,
        "gpt-4o-mini": 0.15 / 1_000_000,
    }
    COST_PER_COMPLETION_TOKEN = {
        "gpt-35-turbo": 2 / 1_000_000,
        "gpt-4-turbo-2024-04-09": 30 / 1_000_000,
        "gpt-4o": 10 / 1_000_000,
        "gpt-4o-mini": 0.60 / 1_000_000,
    }

    MODELS_WITH_TOKEN_PROBS = set(
        [
            "gpt-35-turbo",
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini"
        ]
    )

    MODELS_WITH_STRUCTURED_OUTPUTS = set(
        [
            "gpt-4o-mini",
            "gpt-4o",
        ],
    )

    ERROR_TYPE_MAPPING = {
        "context_length_exceeded": ErrorType.CONTEXT_LENGTH_ERROR,
        "rate_limit_exceeded": ErrorType.RATE_LIMIT_ERROR,
    }

    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        super().__init__(config, cache, tokenizer)
        try:
            from openai import AzureOpenAI
            import tiktoken
        except ImportError:
            raise ImportError(
                "openai is required to use the AzureOpenAILLM. Please install it with: pip install 'refuel-autolabel[openai]'"
            )

        self.tiktoken = tiktoken
        self.model_name = config.model_name() or self.DEFAULT_MODEL
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}

        required_env_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION"
        ]
        for var in required_env_vars:
            if os.getenv(var) is None:
                raise ValueError(f"{var} environment variable not set")

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.llm = partial(
            self.client.chat.completions.create,
            model=self.model_name,
            **self.model_params
        )


    def _label(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        generations = []
        errors = []
        latencies = []
        for prompt in prompts:
            content = [{"type": "text", "text": prompt}]
            start_time = time()
            try:
                if (
                    output_schema is not None
                    and self.model_name in self.MODELS_WITH_STRUCTURED_OUTPUTS
                ):
                    result = self.llm(
                        messages=[{"role": "user", "content": content}],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "response_format",
                                "schema": output_schema,
                                "strict": True,
                            },
                        },
                    )
                else:
                    result = self.llm(
                        messages=[{"role": "user", "content": content}],
                    )

                generations.append(
                    [
                        Generation(
                            text=result.choices[0].message.content,
                            generation_info=None,
                        ),
                    ],
                )
                errors.append(None)
            except Exception as e:
                logger.error(f"Error generating label: {e}")
                generations.append(
                    [
                        Generation(
                            text="",
                            generation_info=None,
                        ),
                    ],
                )
                errors.append(
                    LabelingError(
                        error_type=ErrorType.LLM_PROVIDER_ERROR, error_message=str(e),
                    ),
                )
            end_time = time()
            latencies.append(end_time - start_time)

        return RefuelLLMResult(
            generations=generations,
            errors=errors,
            latencies=latencies,
        )

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        encoding = self.tiktoken.encoding_for_model(self.model_name)
        num_prompt_tokens = len(encoding.encode(prompt))
        
        if label:
            num_completion_tokens = len(encoding.encode(label))
        else:
            num_completion_tokens = self.model_params["max_tokens"]

        return (
            num_prompt_tokens * self.COST_PER_PROMPT_TOKEN[self.model_name] +
            num_completion_tokens * self.COST_PER_COMPLETION_TOKEN[self.model_name]
        )

    def returns_token_probs(self) -> bool:
        return (
            self.model_name is not None
            and self.model_name in self.MODELS_WITH_TOKEN_PROBS
        )

    def get_num_tokens(self, prompt: str) -> int:
        encoding = self.tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(prompt))

