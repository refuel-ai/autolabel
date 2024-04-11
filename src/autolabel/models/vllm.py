import json
import os
import requests
import logging
from typing import List, Optional, Tuple
from time import time

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache
from autolabel.schema import LabelingError, ErrorType, RefuelLLMResult

from langchain.schema import Generation

logger = logging.getLogger(__name__)


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
        try:
            from vllm import LLM, SamplingParams
            from vllm.model_executor.parallel_utils.parallel_state import (
                destroy_model_parallel,
            )
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "vllm is required to use the vllm LLM. Please install it with the following command: pip install vllm"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name())
        self.destroy_model_parallel = destroy_model_parallel
        self.params = SamplingParams(
            max_tokens=self.DEFAULT_PARAMS["max_tokens"],
            temperature=self.DEFAULT_PARAMS["temperature"],
            top_p=self.DEFAULT_PARAMS["top_p"],
        )
        self.model_name = self.config.model_name()
        self.llm = LLM(
            model=self.config.model_name(),
            tensor_parallel_size=self.config.model_params().get(
                "tensor_parallel_size", 1
            ),
        )

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        generations = []
        errors = []
        latencies = []
        for prompt in prompts:
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.llm.generate(
                    prompt_token_ids=[self.tokenizer.apply_chat_template(messages)],
                    sampling_params=self.params,
                    use_tqdm=False,
                )
                generations.append(
                    [
                        Generation(
                            text=response[0].outputs[0].text.strip(),
                        )
                    ]
                )
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
        return RefuelLLMResult(
            generations=generations, errors=errors, latencies=latencies
        )

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0

    def returns_token_probs(self) -> bool:
        return True

    def get_num_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))
