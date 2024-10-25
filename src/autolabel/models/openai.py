import ast
import logging
import os
from functools import cached_property
from time import time
from typing import Dict, List, Optional

from langchain.schema import Generation, HumanMessage, LLMResult
from transformers import AutoTokenizer

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import ErrorType, LabelingError, RefuelLLMResult

logger = logging.getLogger(__name__)


class OpenAILLM(BaseModel):
    CHAT_ENGINE_MODELS = set(
        [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
        ]
    )
    MODELS_WITH_TOKEN_PROBS = set(
        [
            "text-curie-001",
            "text-davinci-003",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
        ]
    )

    SUPPORTS_JSON_OUTPUTS = set(
        [
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
        ]
    )

    SUPPORTS_STRUCTURED_OUTPUTS = set(
        [
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
        ]
    )

    # Default parameters for OpenAILLM
    DEFAULT_MODEL = "gpt-3.5-turbo"
    DEFAULT_PARAMS_COMPLETION_ENGINE = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "model_kwargs": {"logprobs": 1},
        "request_timeout": 30,
    }
    DEFAULT_PARAMS_CHAT_ENGINE = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 1,
        "request_timeout": 30,
    }
    # Reference: https://openai.com/pricing
    COST_PER_PROMPT_TOKEN = {
        "text-davinci-003": 0.02 / 1000,
        "text-curie-001": 0.002 / 1000,
        "gpt-3.5-turbo": 0.0015 / 1000,
        "gpt-3.5-turbo-0301": 0.0015 / 1000,
        "gpt-3.5-turbo-0613": 0.0015 / 1000,
        "gpt-3.5-turbo-16k": 0.003 / 1000,
        "gpt-3.5-turbo-16k-0613": 0.003 / 1000,
        "gpt-4": 0.03 / 1000,
        "gpt-4-0613": 0.03 / 1000,
        "gpt-4-32k": 0.06 / 1000,
        "gpt-4-32k-0613": 0.06 / 1000,
        "gpt-4-0314": 0.03 / 1000,
        "gpt-4-32k-0314": 0.06 / 1000,
        "gpt-4-1106-preview": 0.01 / 1000,
        "gpt-4-0125-preview": 0.01 / 1000,
        "gpt-4o": 0.005 / 1000,
        "gpt-4o-2024-08-06": 0.0025 / 1000,
        "gpt-4o-mini": 0.15 / 1_000_000,
    }
    COST_PER_COMPLETION_TOKEN = {
        "text-davinci-003": 0.02 / 1000,
        "text-curie-001": 0.002 / 1000,
        "gpt-3.5-turbo": 0.002 / 1000,
        "gpt-3.5-turbo-0301": 0.002 / 1000,
        "gpt-3.5-turbo-0613": 0.002 / 1000,
        "gpt-3.5-turbo-16k": 0.004 / 1000,
        "gpt-3.5-turbo-16k-0613": 0.004 / 1000,
        "gpt-4": 0.06 / 1000,
        "gpt-4-0613": 0.06 / 1000,
        "gpt-4-32k": 0.12 / 1000,
        "gpt-4-32k-0613": 0.12 / 1000,
        "gpt-4-0314": 0.06 / 1000,
        "gpt-4-32k-0314": 0.12 / 1000,
        "gpt-4-1106-preview": 0.03 / 1000,
        "gpt-4-0125-preview": 0.03 / 1000,
        "gpt-4o": 0.015 / 1000,
        "gpt-4o-2024-08-06": 0.01 / 1000,
        "gpt-4o-mini": 0.60 / 1_000_000,
    }
    ERROR_TYPE_MAPPING = {
        "context_length_exceeded": ErrorType.CONTEXT_LENGTH_ERROR,
        "rate_limit_exceeded": ErrorType.RATE_LIMIT_ERROR,
    }

    @cached_property
    def _engine(self) -> str:
        if self.model_name is not None and self.model_name in self.CHAT_ENGINE_MODELS:
            return "chat"
        else:
            return "completion"

    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        super().__init__(config, cache, tokenizer)
        try:
            import tiktoken
            from langchain_openai import ChatOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "openai is required to use the OpenAILLM. Please install it with the following command: pip install 'refuel-autolabel[openai]'"
            )
        self.tiktoken = tiktoken
        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL

        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # populate model params and initialize the LLM
        model_params = config.model_params()
        if self._engine == "chat":
            self.model_params = {**self.DEFAULT_PARAMS_CHAT_ENGINE, **model_params}
            self.llm = ChatOpenAI(
                model_name=self.model_name, verbose=False, **self.model_params
            )
        else:
            self.model_params = {
                **self.DEFAULT_PARAMS_COMPLETION_ENGINE,
                **model_params,
            }
            self.llm = OpenAI(
                model_name=self.model_name, verbose=False, **self.model_params
            )

    def _chat_backward_compatibility(
        self, generations: List[LLMResult]
    ) -> List[LLMResult]:
        for generation_options in generations:
            for curr_generation in generation_options:
                generation_info = curr_generation.generation_info
                new_logprobs = {"top_logprobs": []}
                for curr_token in generation_info["logprobs"]["content"]:
                    new_logprobs["top_logprobs"].append(
                        {curr_token["token"]: curr_token["logprob"]}
                    )
                curr_generation.generation_info["logprobs"] = new_logprobs
        return generations

    async def _alabel(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        try:
            start_time = time()
            if self._engine == "chat":
                prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
                generations = None
                if (
                    output_schema is not None
                    and self.model_name in self.SUPPORTS_STRUCTURED_OUTPUTS
                ):
                    result = await self.llm.agenerate(
                        prompts,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "response_format",
                                "schema": output_schema,
                                "strict": True,
                            },
                        },
                    )
                elif (
                    output_schema is not None
                    and self.model_name in self.SUPPORTS_JSON_OUTPUTS
                ):
                    result = await self.llm.agenerate(
                        prompts,
                        response_format={"type": "json_object"},
                    )
                else:
                    logger.info(
                        "Not using structured output despite output_schema provided"
                    )
                    result = await self.llm.agenerate(prompts)
                generations = self._chat_backward_compatibility(result.generations)
            else:
                result = await self.llm.agenerate(prompts)
                generations = result.generations
            end_time = time()
            return RefuelLLMResult(
                generations=generations,
                errors=[None] * len(generations),
                latencies=[end_time - start_time] * len(generations),
            )
        except Exception as e:
            logger.exception(f"Unable to generate prediction: {e}")
            error_message = str(e)
            error_type = ErrorType.LLM_PROVIDER_ERROR
            try:
                json_start, json_end = error_message.find("{"), error_message.rfind("}")
                error_json = ast.literal_eval(error_message[json_start : json_end + 1])[
                    "error"
                ]
                error_code = error_json.get("code")
                error_type = self.ERROR_TYPE_MAPPING.get(
                    error_code, ErrorType.LLM_PROVIDER_ERROR
                )
                error_message = error_json.get("message")
            except Exception as e:
                logger.error(f"Unable to parse OpenAI error message: {e}")

            return RefuelLLMResult(
                generations=[[Generation(text="")] for _ in prompts],
                errors=[
                    LabelingError(
                        error_type=error_type,
                        error_message=error_message,
                    )
                    for _ in prompts
                ],
                latencies=[0 for _ in prompts],
            )

    def _label(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        try:
            start_time = time()
            if self._engine == "chat":
                prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
                if (
                    output_schema is not None
                    and self.model_name in self.SUPPORTS_STRUCTURED_OUTPUTS
                ):
                    result = self.llm.generate(
                        prompts,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "response_format",
                                "schema": output_schema,
                                "strict": True,
                            },
                        },
                    )
                elif (
                    output_schema is not None
                    and self.model_name in self.SUPPORTS_JSON_OUTPUTS
                ):
                    result = self.llm.generate(
                        prompts,
                        response_format={"type": "json_object"},
                    )
                else:
                    logger.info(
                        "Not using structured output despite output_schema provided"
                    )
                    result = self.llm.generate(prompts)
                generations = self._chat_backward_compatibility(result.generations)
            else:
                result = self.llm.generate(prompts)
                generations = result.generations
            end_time = time()
            return RefuelLLMResult(
                generations=generations,
                errors=[None] * len(generations),
                latencies=[end_time - start_time] * len(generations),
            )
        except Exception as e:
            logger.exception(f"Unable to generate prediction: {e}")

            error_message = str(e)
            error_type = ErrorType.LLM_PROVIDER_ERROR
            try:
                json_start, json_end = error_message.find("{"), error_message.rfind("}")
                error_json = ast.literal_eval(error_message[json_start : json_end + 1])[
                    "error"
                ]
                error_code = error_json.get("code")
                error_type = self.ERROR_TYPE_MAPPING.get(
                    error_code, ErrorType.LLM_PROVIDER_ERROR
                )
                error_message = error_json.get("message")
            except Exception as e:
                logger.error(f"Unable to parse OpenAI error message: {e}")
            return RefuelLLMResult(
                generations=[[Generation(text="")] for _ in prompts],
                errors=[
                    LabelingError(
                        error_type=error_type,
                        error_message=error_message,
                    )
                    for _ in prompts
                ],
                latencies=[0 for _ in prompts],
            )

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        encoding = self.tiktoken.encoding_for_model(self.model_name)
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

    def get_num_tokens(self, prompt: str) -> int:
        encoding = self.tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(prompt))
