from typing import List, Optional

from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.cache import BaseCache
from autolabel.schema import RefuelLLMResult
from autolabel.schema import (
    GenerationCacheEntry,
    LabelingError,
    RefuelLLMResult,
    ErrorType,
)
from langchain.schema import HumanMessage, Generation
import logging

logger = logging.getLogger(__name__)


class LiteLLM(BaseModel):
    SEP_REPLACEMENT_TOKEN = "@@"
    DEFAULT_MODEL = "gpt-3.5-turbo"
    DEFAULT_PARAMS = {
        "max_tokens_to_sample": 1000,
        "temperature": 0.0,
    }

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)

        try:
            import litellm
            from litellm import completion, completion_cost, batch_completion
        except ImportError:
            raise ImportError(
                "anthropic is required to use the anthropic LLM. Please install it with the following command: pip install 'refuel-autolabel[anthropic]'"
            )

        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL
        # populate model params
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}
        self.completion_cost = completion_cost
        self.completion = completion
        self.batch_completion = batch_completion

    def _label_individually(self, prompts: List[str]) -> RefuelLLMResult:
        """Label each prompt individually. Should be used only after trying as a batch first.

        Args:
            prompts (List[str]): List of prompts to label

        Returns:
            LLMResult: LLMResult object with generations
            List[LabelingError]: List of errors encountered while labeling
        """
        generations = []
        errors = []
        for prompt in prompts:
            try:
                messages = [{"role": "user", "content": prompt}]
                custom_model_name = "claude-instant-1"  # [TEST Variable] - using to debug an issue in testing
                response = self.completion(model=custom_model_name, messages=messages)
                generations.append(
                    Generation(text=response["choices"][0]["message"]["content"])
                )
                errors.append(None)
            except Exception as e:
                print(f"Error generating from LLM: {e}")
                print(f"self.model_name: {self.model_name}")
                generations.append([Generation(text="")])
                errors.append(
                    LabelingError(
                        error_type=ErrorType.LLM_PROVIDER_ERROR, error_message=str(e)
                    )
                )

        return RefuelLLMResult(generations=generations, errors=errors)

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
        try:
            custom_model_name = "claude-instant-1"  # [TEST Variable] - using to debug an issue in testing
            results = self.batch_completion(model=custom_model_name, messages=prompts)
            # translate to RefuelLLMResult Generations type
            result_generations = [
                Generation(text=result["choices"][0]["message"]["content"])
                for result in results
            ]
            return RefuelLLMResult(
                generations=result_generations, errors=[None] * len(result_generations)
            )
        except Exception as e:
            return self._label_individually(prompts)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        # total_cost = self.completion_cost(model=self.model_name, prompt=prompt, completion=label)
        return 0  # [TEST Variable] - using to debug an issue in testing

    def returns_token_probs(self) -> bool:
        return False
