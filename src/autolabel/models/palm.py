from functools import cached_property
from typing import List, Optional
import logging

from langchain.chat_models import ChatVertexAI
from langchain.llms import VertexAI
from langchain.schema import LLMResult, HumanMessage, Generation

from autolabel.models import BaseModel
from autolabel.configs import AutolabelConfig
from autolabel.cache import BaseCache

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class PaLMLLM(BaseModel):
    SEP_REPLACEMENT_TOKEN = "@@"
    CHAT_ENGINE_MODELS = ["chat-bison@001"]

    DEFAULT_MODEL = "text-bison@001"
    DEFAULT_PARAMS = {"temperature": 0}

    # Reference: https://cloud.google.com/vertex-ai/pricing
    COST_PER_CHARACTER = {
        "text-bison@001": 0.001 / 1000,
        "chat-bison@001": 0.0005 / 1000,
        "textembedding-gecko@001": 0.0001 / 1000,
    }

    @cached_property
    def _engine(self) -> str:
        if self.model_name is not None and self.model_name in self.CHAT_ENGINE_MODELS:
            return "chat"
        else:
            return "completion"

    def __init__(self, config: AutolabelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)
        # populate model name
        self.model_name = config.model_name() or self.DEFAULT_MODEL

        # populate model params and initialize the LLM
        model_params = config.model_params()
        self.model_params = {
            **self.DEFAULT_PARAMS,
            **model_params,
        }
        if self._engine == "chat":
            self.llm = ChatVertexAI(model_name=self.model_name, **self.model_params)
        else:
            self.llm = VertexAI(model_name=self.model_name, **self.model_params)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _label_with_retry(self, prompts: List[str]) -> LLMResult:
        return self.llm.generate(prompts)

    def _label(self, prompts: List[str]) -> LLMResult:
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
            result = self._label_with_retry(prompts)
            for generations in result.generations:
                for generation in generations:
                    generation.text = generation.text.replace(
                        self.SEP_REPLACEMENT_TOKEN, "\n"
                    )
            return result
        except Exception as e:
            logger.error(f"Error generating from LLM: {e}.")
        generations = [[Generation(text="")] for _ in prompts]
        return LLMResult(generations=generations)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        if self.model_name is None:
            return 0.0
        cost_per_char = self.COST_PER_CHARACTER.get(self.model_name, 0.0)
        return cost_per_char * len(prompt)

    def returns_token_probs(self) -> bool:
        return False
