"""Base interface that all model providers will implement."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple

from langchain.schema import LLMResult

from autolabel.configs import AutolabelConfig
from autolabel.schema import CacheEntry
from autolabel.cache import BaseCache


class BaseModel(ABC):
    def __init__(self, config: AutolabelConfig, cache: BaseCache) -> None:
        self.config = config
        self.cache = cache
        self.model_params = config.model_params()
        # Specific classes that implement this interface should run initialization steps here
        # E.g. initializing the LLM model with required parameters from ModelConfig

    def label(self, prompts: List[str]) -> Tuple[LLMResult, float]:
        """Label a list of prompts."""
        existing_prompts = {}
        missing_prompt_idxs = list(range(len(prompts)))
        missing_prompts = prompts
        llm_output = {}
        cost = 0.0
        if self.cache:
            (
                existing_prompts,
                missing_prompt_idxs,
                missing_prompts,
            ) = self.get_cached_prompts(prompts)

        # label missing prompts
        if len(missing_prompts) > 0:
            new_results = self._label(missing_prompts)
            for ind, prompt in enumerate(missing_prompts):
                cost += self.get_cost(
                    prompt, label=new_results.generations[ind][0].text
                )

            # Set the existing prompts to the new results
            for i, result in zip(missing_prompt_idxs, new_results.generations):
                existing_prompts[i] = result

            if self.cache:
                self.update_cache(missing_prompt_idxs, new_results, prompts)

            llm_output = new_results.llm_output

        generations = [existing_prompts[i] for i in range(len(prompts))]
        return LLMResult(generations=generations, llm_output=llm_output), cost

    @abstractmethod
    def _label(self, prompts: List[str]) -> LLMResult:
        # TODO: change return type to do parsing in the Model class
        pass

    @abstractmethod
    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
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
            cache_entry = CacheEntry(
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

        for i, result in zip(missing_prompt_idxs, new_results.generations):
            # If the result is empty, don't cache it
            # This result was likely produced due to an error
            if result[0].text == "":
                continue

            cache_entry = CacheEntry(
                prompt=prompts[i],
                model_name=self.model_name,
                model_params=model_params_string,
                generations=result,
            )
            self.cache.update(cache_entry)

    @abstractmethod
    def returns_token_probs(self) -> bool:
        """Whether the LLM supports returning logprobs of generated tokens

        Returns:
            bool: whether the LLM returns supports returning logprobs of generated tokens
        """
        pass
