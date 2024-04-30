import json
import os
import requests
import logging
from typing import List, Optional, Tuple
from time import time


import gc
import torch

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache
from autolabel.schema import LabelingError, ErrorType, RefuelLLMResult

from langchain.schema import Generation

logger = logging.getLogger(__name__)


class VLLMModel(BaseModel):
    DEFAULT_PARAMS = {
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1.0,
    }

    def __init__(
        self,
        config: AutolabelConfig,
        cache: BaseCache = None,
    ) -> None:
        super().__init__(config, cache)
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "vllm is required to use the vllm LLM. Please install it with the following command: pip install vllm"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name())
        self.params = SamplingParams(
            max_tokens=self.DEFAULT_PARAMS["max_tokens"],
            temperature=self.DEFAULT_PARAMS["temperature"],
            top_p=self.DEFAULT_PARAMS["top_p"],
            use_beam_search=True,
            best_of=5,
            logprobs=1,
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
                tokenized_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                if len(tokenized_prompt) > 4096:
                    logger.warning(
                        f"Input is greater than 4096 tokens: {len(tokenized_prompt)}"
                    )
                response = self.llm.generate(
                    prompt_token_ids=[tokenized_prompt],
                    sampling_params=self.params,
                    use_tqdm=False,
                )
                generations.append(
                    [
                        Generation(
                            text=response[0]
                            .outputs[0]
                            .text.strip()
                            .replace("<|eot_id|>", ""),
                            generation_info=(
                                {
                                    "logprobs": {
                                        "top_logprobs": self._process_confidence_request(
                                            response[0].outputs[0].logprobs
                                        )
                                    }
                                }
                                if self.config.confidence()
                                else None
                            ),
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

    def _process_confidence_request(self, logprobs):
        resp = []
        for item in logprobs:
            key = list(item.keys())[0]
            curr_logprob_obj = item[key]
            resp.append({curr_logprob_obj.decoded_token: curr_logprob_obj.logprob})
        return resp

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        return 0

    def returns_token_probs(self) -> bool:
        return True

    def get_num_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))

    def destroy(self):
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
