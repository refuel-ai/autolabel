"""Base interface that all model providers will implement."""

from abc import ABC, abstractmethod
from time import time
from typing import Dict, List, Optional

from langchain.schema import Generation

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.schema import (
    GenerationCacheEntry,
    RefuelLLMResult,
)


class BaseModel(ABC):
    TTL_MS = 60 * 60 * 24 * 7 * 1000  # 1 week
    DEFAULT_CONTEXT_LENGTH = None

    def __init__(self, config: AutolabelConfig, cache: BaseCache) -> None:
        self.config = config
        self.cache = cache
        self.model_params = config.model_params()
        self.max_context_length = config.max_context_length(
            default=self.DEFAULT_CONTEXT_LENGTH
        )
        # Specific classes that implement this interface should run initialization steps here
        # E.g. initializing the LLM model with required parameters from ModelConfig

    async def label(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        """Label a list of prompts."""
        existing_prompts = {}
        missing_prompt_idxs = list(range(len(prompts)))
        missing_prompts = prompts
        costs = []
        errors = [None for i in range(len(prompts))]
        latencies = [0 for i in range(len(prompts))]
        if self.cache:
            (
                existing_prompts,
                missing_prompt_idxs,
                missing_prompts,
            ) = self.get_cached_prompts(prompts)
        # label missing prompts
        if len(missing_prompts) > 0:
            if hasattr(self, "_alabel"):
                new_results = await self._alabel(missing_prompts, output_schema)
            else:
                new_results = self._label(missing_prompts, output_schema)
            for ind, prompt in enumerate(missing_prompts):
                costs.append(
                    self.get_cost(
                        prompt,
                        label=new_results.generations[ind][0].text,
                        llm_output=new_results.llm_output,
                    )
                )

            # Set the existing prompts to the new results
            for i, result, error, latency in zip(
                missing_prompt_idxs,
                new_results.generations,
                new_results.errors,
                new_results.latencies,
            ):
                existing_prompts[i] = result
                errors[i] = error
                latencies[i] = latency

            if self.cache:
                self.update_cache(missing_prompt_idxs, new_results, prompts)
        generations = [existing_prompts[i] for i in range(len(prompts))]
        return RefuelLLMResult(
            generations=generations, costs=costs, errors=errors, latencies=latencies
        )

    @abstractmethod
    def _label(self, prompts: List[str], output_schema: Dict) -> RefuelLLMResult:
        # TODO: change return type to do parsing in the Model class
        pass

    @abstractmethod
    def get_cost(
        self, prompt: str, label: Optional[str] = "", llm_output: Optional[Dict] = None
    ) -> float:
        pass

    def get_cached_prompts(self, prompts: List[str]) -> Optional[str]:
        """Get prompts that are already cached."""
        model_params_string = str(
            sorted([(k, v) for k, v in self.model_params.items()])
        )
        missing_prompts = []
        missing_prompt_idxs = []
        existing_prompts = {}
        for i, prompt in enumerate(prompts):
            cache_entry = GenerationCacheEntry(
                prompt=prompt,
                model_name=self.model_name,
                model_params=model_params_string,
            )
            cache_val = self.cache.lookup(cache_entry)
            if cache_val:
                existing_prompts[i] = cache_val
            else:
                missing_prompts.append(prompt)
                missing_prompt_idxs.append(i)
        return (
            existing_prompts,
            missing_prompt_idxs,
            missing_prompts,
        )

    def update_cache(self, missing_prompt_idxs, new_results, prompts):
        """Update the cache with new results."""
        model_params_string = str(
            sorted([(k, v) for k, v in self.model_params.items()])
        )

        for i, result, error in zip(
            missing_prompt_idxs, new_results.generations, new_results.errors
        ):
            # If there was an error, don't cache the result
            if error is not None:
                continue

            cache_entry = GenerationCacheEntry(
                prompt=prompts[i],
                model_name=self.model_name,
                model_params=model_params_string,
                generations=result,
                ttl_ms=self.TTL_MS,
            )
            self.cache.update(cache_entry)

    @abstractmethod
    def returns_token_probs(self) -> bool:
        """Whether the LLM supports returning logprobs of generated tokens

        Returns:
            bool: whether the LLM returns supports returning logprobs of generated tokens
        """
        pass

    @abstractmethod
    def get_num_tokens(self, prompt: str) -> int:
        """
        Get the number of tokens in the prompt"""
        pass
