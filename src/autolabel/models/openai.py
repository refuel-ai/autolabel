import logging
import os
from functools import cached_property
from time import time
from typing import List, Optional

from langchain.schema import HumanMessage, LLMResult

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import RefuelLLMResult

logger = logging.getLogger(__name__)


class OpenAILLM(BaseModel):
    CHAT_ENGINE_MODELS = [
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
    ]
    MODELS_WITH_TOKEN_PROBS = [
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
    ]
    JSON_MODE_MODELS = [
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo",
        "gpt-4-0125-preview",
        "gpt-4-1106-preview",
        "gpt-4-turbo-preview",
    ]

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
        "request_timeout": 30,
    }
    DEFAULT_QUERY_PARAMS_CHAT_ENGINE = {"logprobs": True, "top_logprobs": 1}

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
    }

    @cached_property
    def _engine(self) -> str:
        if self.model_name is not None and self.model_name in self.CHAT_ENGINE_MODELS:
            return "chat"
        else:
            return "completion"

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)
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
        if config.logit_bias() != 0:
            model_params = {
                **model_params,
                **self._generate_logit_bias(),
            }

        self.query_params = {}
        if self._engine == "chat":
            self.model_params = {**self.DEFAULT_PARAMS_CHAT_ENGINE, **model_params}
            self.query_params = self.DEFAULT_QUERY_PARAMS_CHAT_ENGINE
            self.llm = ChatOpenAI(
                model_name=self.model_name, verbose=False, **self.model_params
            )
            if config.json_mode():
                if self.model_name not in self.JSON_MODE_MODELS:
                    logger.warning(
                        f"json_mode is not supported for model {self.model_name}. Disabling json_mode."
                    )
                else:
                    self.query_params["response_format"] = {"type": "json_object"}
        else:
            self.model_params = {
                **self.DEFAULT_PARAMS_COMPLETION_ENGINE,
                **model_params,
            }
            self.llm = OpenAI(
                model_name=self.model_name, verbose=False, **self.model_params
            )

    def _generate_logit_bias(self) -> None:
        """Generates logit bias for the labels specified in the config

        Returns:
            Dict: logit bias and max tokens
        """
        if len(self.config.labels_list()) == 0:
            logger.warning(
                "No labels specified in the config. Skipping logit bias generation."
            )
            return {}
        encoding = self.tiktoken.encoding_for_model(self.model_name)
        logit_bias = {}
        max_tokens = 0
        for label in self.config.labels_list():
            if label not in logit_bias:
                tokens = encoding.encode(label)
                for token in tokens:
                    logit_bias[token] = self.config.logit_bias()
                max_tokens = max(max_tokens, len(tokens))
        logit_bias[encoding.eot_token] = self.config.logit_bias()
        return {"logit_bias": logit_bias, "max_tokens": max_tokens}

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

    async def _alabel(self, prompts: List[str]) -> RefuelLLMResult:
        start_time = time()
        if self._engine == "chat":
            prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
            result = await self.llm.agenerate(prompts, **self.query_params)
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

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        start_time = time()
        if self._engine == "chat":
            prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
            result = self.llm.generate(prompts, **self.query_params)
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
