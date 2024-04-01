import os
import logging
from functools import cached_property
from time import time
from typing import List, Optional

from langchain.schema import Generation, HumanMessage, LLMResult
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from autolabel.cache import BaseCache
from autolabel.configs import AutolabelConfig
from autolabel.models import BaseModel
from autolabel.schema import ErrorType, LabelingError, RefuelLLMResult

logger = logging.getLogger(__name__)


class GoogleLLM(BaseModel):
    SEP_REPLACEMENT_TOKEN = "@@"
    CHAT_ENGINE_MODELS = ["gemini-pro"]

    DEFAULT_MODEL = "gemini-pro"

    # Reference: https://ai.google.dev/pricing
    COST_PER_PROMPT_CHARACTER = {
        "gemini-pro": 0.000125 / 1000,
    }

    COST_PER_COMPLETION_CHARACTER = {
        "gemini-pro": 0.000375 / 1000,
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
    ) -> None:
        try:
            import tiktoken
            from vertexai.generative_models import (
                GenerativeModel,
                HarmBlockThreshold,
                HarmCategory,
            )
        except ImportError:
            raise ImportError(
                "tiktoken and langchain_google_vertexai are requried for google models. Please install it with the following command: pip install 'refuel-autolabel[google]'"
            )

        self.DEFAULT_PARAMS = {
            "temperature": 0.0,
            "top_k": 3,
        }

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        super().__init__(config, cache)

        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL
        # populate model params and initialize the LLM
        model_params = config.model_params()
        self.model_params = {**self.DEFAULT_PARAMS, **model_params}

        if self._engine == "chat":
            self.llm = GenerativeModel(
                model_name=self.model_name
            )
        else:
            self.llm = GenerativeModel(
                model_name=self.model_name, **self.model_params
            )
        self.tiktoken = tiktoken

    # async def _alabel(self, prompts: List[str]) -> RefuelLLMResult:
    #     generations = []
    #     errors = []
    #     latencies = []
    #     for prompt in prompts:
    #         try:
    #             start_time = time()
    #             response = self.llm.generate_content(
    #                 prompt,
    #                 generation_config=self.DEFAULT_PARAMS,
    #                 safety_settings=self.safety_settings,
    #             )
    #             end_time = time()
    #             latency = end_time - start_time
    #             generations.append(
    #                 [
    #                     Generation(
    #                         text=response.text,
    #                         generation_info=(
    #                             {"logprobs": {"top_logprobs": response["logprobs"]}}
    #                             if self.config.confidence()
    #                             else None
    #                         ),
    #                     )
    #                 ]
    #             )
    #             errors.append(None)
    #             latencies.append(latency)
    #         except Exception as e:
    #             # This signifies an error in generating the response using RefuelLLm
    #             logger.error(
    #                 f"Unable to generate prediction: {e}",
    #             )
    #             generations.append([Generation(text="")])
    #             errors.append(
    #                 LabelingError(
    #                     error_type=ErrorType.LLM_PROVIDER_ERROR, error_message=str(e)
    #                 )
    #             )
    #             latencies.append(0)
    #     return RefuelLLMResult(
    #         generations=generations, errors=errors, latencies=latencies
    #     )

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        generations = []
        errors = []
        latencies = []
        for prompt in prompts:
            try:
                start_time = time()
                response = self.llm.generate_content(
                    prompt,
                    generation_config=self.DEFAULT_PARAMS,
                    safety_settings=self.safety_settings,
                )
                end_time = time()
                latency = end_time - start_time
                generations.append(
                    [
                        Generation(
                            text=response.text,
                        )
                    ]
                )
                errors.append(None)
                latencies.append(latency)
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
        if self.model_name is None:
            return 0.0
        cost_per_prompt_char = self.COST_PER_PROMPT_CHARACTER[self.model_name]
        cost_per_completion_char = self.COST_PER_COMPLETION_CHARACTER[self.model_name]
        return cost_per_prompt_char * len(prompt) + cost_per_completion_char * (
            len(label) if label else 0.0
        )

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        return super().get_num_tokens(prompt)
