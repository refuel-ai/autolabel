import json
import os
import requests
import logging
from typing import List, Optional, Tuple
from time import time
from vllm import LLM, SamplingParams

import gc
import torch
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache
from autolabel.schema import LabelingError, ErrorType, RefuelLLMResult

from langchain.schema import Generation

UNRETRYABLE_ERROR_CODES = [400, 422]
logger = logging.getLogger(__name__)


class UnretryableError(Exception):
    """This is an error which is unretriable from autolabel."""


class VLLMModel(BaseModel):
    DEFAULT_PARAMS = {
        "max_tokens": 128,
        "temperature": 0.05,
        "top_p": 0.95,
    }

    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
    ) -> None:
        super().__init__(config, cache)
        self.params = SamplingParams(
            max_tokens=self.DEFAULT_PARAMS["max_tokens"],
            temperature=self.DEFAULT_PARAMS["temperature"],
            top_p=self.DEFAULT_PARAMS["top_p"],
        )
        self.llm = LLM(model=self.config.model_name())
        self.model_name = self.config.model_name()

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        generations = []
        errors = []
        latencies = []
        for prompt in prompts:
            try:
                print(prompt)
                response = self.llm.generate(prompt, self.params, use_tqdm=False)
                generations.append(
                    [Generation(
                        text=response[0].outputs[0].text.strip(),
                    )]
                )
                print(response[0].outputs[0].text.strip())
                errors.append(None)
                latencies.append(0)
            except Exception as e:
                # This signifies an error in generating the response using RefuelLLm
                logger.error(
                    f"Unable to generate prediction: {e}",
                )
                generations.append([Generation(text="")])
                errors.append(
                    LabelingError(
                        error_type=ErrorType.LLM_PROVIDER_ERROR, error_message=str(e)
                    )
                )
                latencies.append(0)
        return RefuelLLMResult(generations=generations, errors=errors, latencies = latencies)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        return len(prompt)
    
    def destroy(self):
        destroy_model_parallel()
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()