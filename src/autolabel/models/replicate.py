from typing import List, Optional
from time import time
import logging
import requests

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache
from autolabel.schema import RefuelLLMResult


import os

logger = logging.getLogger(__name__)


class ReplicateLLM(BaseModel):
    REPLICATE_MAINTAINED_MODELS = [
        "meta/llama-2-70b",
        "meta/llama-2-13b",
        "meta/llama-2-7b",
        "meta/llama-2-70b-chat",
        "meta/llama-2-13b-chat",
        "meta/llama-2-7b-chat",
        "mistralai/mistral-7b-v0.1",
        "mistralai/mistral-7b-instruct-vo.2",
        "mistralai/mixtral-8x7b-instruct-v0.1",
    ]

    # Default parameters for OpenAILLM
    DEFAULT_MODEL = "meta/llama-2-7b-chat"

    DEFAULT_PARAMS_COMPLETION_ENGINE = {
        "max_tokens": 1000,
        "temperature": 0.01,
        "model_kwargs": {"logprobs": 1},
        "request_timeout": 30,
    }

    # Reference: https://replicate.com/docs/billing
    COST_PER_PROMPT_TOKEN = {
        "meta/llama-2-70b": 0.65 / 1e6,
        "meta/llama-2-13b": 0.10 / 1e6,
        "meta/llama-2-7b": 0.05 / 1e6,
        "meta/llama-2-70b-chat": 0.65 / 1e6,
        "meta/llama-2-13b-chat": 0.10 / 1e6,
        "meta/llama-2-7b-chat": 0.05 / 1e6,
        "mistralai/mistral-7b-v0.1": 0.05 / 1e6,
        "mistralai/mistral-7b-instruct-v0.2": 0.05 / 1e6,
        "mistralai/mixtral-8x7b-instruct-v0.1": 0.30 / 1e6,
    }
    COST_PER_COMPLETION_TOKEN = {
        "meta/llama-2-70b": 2.75 / 1e6,
        "meta/llama-2-13b": 0.50 / 1e6,
        "meta/llama-2-7b": 0.25 / 1e6,
        "meta/llama-2-70b-chat": 2.75 / 1e6,
        "meta/llama-2-13b-chat": 0.50 / 1e6,
        "meta/llama-2-7b-chat": 0.25 / 1e6,
        "mistralai/mistral-7b-v0.1": 0.25 / 1e6,
        "mistralai/mistral-7b-instruct-v0.2": 0.25 / 1e6,
        "mistralai/mixtral-8x7b-instruct-v0.1": 1.00 / 1e6,
    }

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)
        try:
            from langchain_community.llms import Replicate
            from transformers import LlamaTokenizerFast
        except ImportError:
            raise ImportError(
                "replicate is required to use the ReplicateLLM. Please install it with the following command: pip install 'refuel-autolabel[replicate]'"
            )

        if os.getenv("REPLICATE_API_TOKEN") is None:
            raise ValueError("REPLICATE_API_TOKEN environment variable not set")

        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL

        # populate model params and initialize the LLM
        model_params = config.model_params()

        self.model_params = {
            **self.DEFAULT_PARAMS_COMPLETION_ENGINE,
            **model_params,
        }

        # get latest model version, required by langchain to process replicate generations
        response = requests.get(
            f"https://api.replicate.com/v1/models/{self.model_name}",
            headers={"Authorization": f"Token {os.environ['REPLICATE_API_TOKEN']}"},
        )
        if response.status_code == 404:
            raise ValueError(f"Model {self.model_name} not found on Replicate")
        latest_model_version = response.json()["latest_version"]["id"]

        self.llm = Replicate(
            model=f"{self.model_name}:{latest_model_version}",
            verbose=False,
            **self.model_params,
        )

        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )

    def is_model_managed_by_replicate(self) -> bool:
        return self.model_name in self.REPLICATE_MAINTAINED_MODELS

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        try:
            start_time = time()
            result = self.llm.generate(prompts)
            generations = result.generations
            end_time = time()
            return RefuelLLMResult(
                generations=generations,
                errors=[None] * len(generations),
                latencies=[end_time - start_time] * len(generations),
            )
        except Exception as e:
            return self._label_individually(prompts)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        if self.is_model_mangaed_by_replicate():
            num_prompt_toks = len(self.tokenizer.encode(prompt))
            if label:
                num_label_toks = len(self.tokenizer.encode(label))
            else:
                # get an upper bound
                num_label_toks = self.model_params["max_tokens"]

            cost_per_prompt_token = self.COST_PER_PROMPT_TOKEN[self.model_name]
            cost_per_completion_token = self.COST_PER_COMPLETION_TOKEN[self.model_name]
            return (num_prompt_toks * cost_per_prompt_token) + (
                num_label_toks * cost_per_completion_token
            )
        else:
            # TODO - at the moment it's not possible to calculate it https://github.com/replicate/replicate-python/issues/243
            return 0

    def returns_token_probs(self) -> bool:
        return (
            self.model_name is not None
            and self.model_name in self.MODELS_WITH_TOKEN_PROBS
        )

    def get_num_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))
