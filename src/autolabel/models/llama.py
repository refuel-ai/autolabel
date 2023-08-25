import json
import os
import requests
import logging
from typing import List, Optional

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache
from autolabel.schema import LabelingError, ErrorType, RefuelLLMResult

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)
from langchain.schema import Generation

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch
import os
import sys
from peft import PeftModel, PeftConfig
from transformers import (
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM
)
from vllm import LLM
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class Llama(BaseModel):
    DEFAULT_PARAMS = {
        "max_new_tokens": 128,
        "temperature": 0.05,
        "top_p": 0.9,
    }

    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
    ) -> None:
        super().__init__(config, cache)

        # populate model name
        # This is unused today, but in the future could
        # be used to decide which refuel model is queried
        self.model_name = config.model_name()
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}

        self.labeling_model = LLM(self.model_name, tensor_parallel_size=1)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _label_with_retry(self, prompt: str) -> requests.Response:
        sampling_param = SamplingParams(top_p=self.model_params["top_p"], temperature=self.model_params["temperature"], max_tokens=self.model_params["max_new_tokens"])
        outputs = self.labeling_model.generate(prompt, sampling_params=sampling_param)
        return outputs[0].outputs[0].text

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        generations = []
        errors = []
        for prompt in prompts:
            try:
                response = self._label_with_retry(prompt)
                generations.append([Generation(text=response)])
                errors.append(None)
            except Exception as e:
                # This signifies an error in generating the response using RefuelLLm
                logger.error(
                    f"Unable to generate prediction: {e}",
                )
                generations.append([Generation(text="")])
                errors.append(
                    LabelingError(error_type=ErrorType.LLM_PROVIDER_ERROR, error=e)
                )
        return RefuelLLMResult(generations=generations, errors=errors)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0

    def returns_token_probs(self) -> bool:
        return False
