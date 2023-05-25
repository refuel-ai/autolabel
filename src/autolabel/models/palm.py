from functools import cached_property
from typing import List, Optional

from langchain.chat_models import ChatVertexAI
from langchain.llms import VertexAI
from langchain.schema import LLMResult, HumanMessage, Generation

from autolabel.models import BaseModel
from autolabel.configs import ModelConfig
from autolabel.cache import BaseCache


class PaLMLLM(BaseModel):
    CHAT_ENGINE_MODELS = ["chat-bison@001"]

    DEFAULT_MODEL = "text-bison@001"
    DEFAULT_PARAMS = {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
    }

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

    def __init__(self, config: ModelConfig, cache: BaseCache = None) -> None:
        super().__init__(config, cache)
        # populate model name
        self.model_name = config.get_model_name() or self.DEFAULT_MODEL

        # populate model params and initialize the LLM
        model_params = config.get_model_params()
        if self._engine == "chat":
            # Doesn't take in any params
            self.llm = ChatVertexAI(model_name=self.model_name)
        else:
            self.model_params = {
                **self.DEFAULT_PARAMS,
                **model_params,
            }
            self.llm = VertexAI(model_name=self.model_name)

    def _label(self, prompts: List[str]) -> LLMResult:
        if self._engine == "chat":
            # Need to convert list[prompts] -> list[messages]
            # Currently the entire prompt is stuck into the "human message"
            # We might consider breaking this up into human vs system message in future
            prompts = [[HumanMessage(content=prompt)] for prompt in prompts]
        try:
            generations = self.llm.generate(prompts)
            print(f"LLM generations: {generations}")
            return generations
        except Exception as e:
            print(f"Error generating from LLM: {e}, returning empty result")
            generations = [[Generation(text="")] for _ in prompts]
            return LLMResult(generations=generations)

    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        if self.model_name is None:
            return 0.0
        cost_per_char = self.COST_PER_CHARACTER.get(self.model_name, 0.0)
        return cost_per_char * len(prompt)
        
    def returns_token_probs(self) -> bool:
        return (
            self.model_name is not None
            and self.model_name in self.MODELS_WITH_TOKEN_PROBS
        )
