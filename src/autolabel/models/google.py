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
    COST_PER_CHARACTER = {
        "gemini-pro": 0.000125 / 1000,
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
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
                HarmBlockThreshold,
                HarmCategory,
            )
        except ImportError:
            raise ImportError(
                "tiktoken and langchain_google_genai. Please install it with the following command: pip install 'refuel-autolabel[google]'"
            )

        self.DEFAULT_PARAMS = {
            "temperature": 0.0,
            "topK": 3,
            "safety_settings": {
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        }

        super().__init__(config, cache)
        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL
        # populate model params and initialize the LLM
        model_params = config.model_params()
        self.model_params = {
            **model_params,
            **self.DEFAULT_PARAMS,
        }
        if self._engine == "chat":
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name, **self.model_params
            )
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name, **self.model_params
            )

        self.tiktoken = tiktoken

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _label_with_retry(self, prompts: List[str]) -> LLMResult:
        start_time = time()
        response = self.llm.generate(prompts)
        return response, time() - start_time

    def _label_individually(self, prompts: List[str]) -> RefuelLLMResult:
        """Label each prompt individually. Should be used only after trying as a batch first.

        Args:
            prompts (List[str]): List of prompts to label

        Returns:
            RefuelLLMResult: RefuelLLMResult object
        """
        generations = []
        errors = []
        latencies = []
        for i, prompt in enumerate(prompts):
            try:
                response, latency = self._label_with_retry([prompt])
                for generation in response.generations[0]:
                    generation.text = generation.text.replace(
                        self.SEP_REPLACEMENT_TOKEN, "\n"
                    )
                generations.append(response.generations[0])
                errors.append(None)
                latencies.append(latency)
            except Exception as e:
                print(f"Error generating from LLM: {e}, returning empty generation")
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

    def _label(self, prompts: List[str]) -> RefuelLLMResult:
        for prompt in prompts:
            if self.SEP_REPLACEMENT_TOKEN in prompt:
                logger.warning(
                    f"""Current prompt contains {self.SEP_REPLACEMENT_TOKEN} 
                                which is currently used as a separator token by refuel
                                llm. It is highly recommended to avoid having any
                                occurences of this substring in the prompt.
                            """
                )
        prompts = [
            prompt.replace("\n", self.SEP_REPLACEMENT_TOKEN) for prompt in prompts
        ]
        if self._engine == "chat":
            # Need to convert list[prompts] -> list[messages]
            # Currently the entire prompt is stuck into the "human message"
            # We might consider breaking this up into human vs system message in future
            prompts = [[HumanMessage(content=prompt)] for prompt in prompts]

        try:
            start_time = time()
            result = self._label_with_retry(prompts)
            end_time = time()
            for generations in result.generations:
                for generation in generations:
                    generation.text = generation.text.replace(
                        self.SEP_REPLACEMENT_TOKEN, "\n"
                    )
            return RefuelLLMResult(
                generations=result.generations,
                errors=[None] * len(result.generations),
                latencies=[end_time - start_time] * len(result.generations),
            )
        except Exception as e:
            return self._label_individually(prompts)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        if self.model_name is None:
            return 0.0
        cost_per_char = self.COST_PER_CHARACTER.get(self.model_name, 0.0)
        return cost_per_char * len(prompt) + cost_per_char * (
            len(label) if label else 1
        )

    def returns_token_probs(self) -> bool:
        return False

    def get_num_tokens(self, prompt: str) -> int:
        # TODO(dhruva): Replace with actual tokenizer once that is available
        encoding = self.tiktoken.encoding_for_model("gpt2")
        return len(encoding.encode(prompt))
