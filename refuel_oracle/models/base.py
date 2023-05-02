"""Base interface that all model providers will implement."""

from abc import ABC, abstractmethod
from typing import List, Optional

from langchain.schema import LLMResult
from refuel_oracle.models import ModelConfig


class BaseModel(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        # Specific classes that implement this interface should run initialization steps here
        # E.g. initializing the LLM model with required parameters from ModelConfig

    @abstractmethod
    def label(self, prompts: List[str]) -> List[LLMResult]:
        # TODO: change return type to do parsing in the Model class
        pass

    @abstractmethod
    def get_cost(self, prompt: str, label: Optional[str] = "") -> float:
        pass
