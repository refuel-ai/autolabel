

"""Base interface for all prediction tasks to expose."""

from abc import ABC, abstractmethod
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import LLMResult

from refuel_oracle.config import Config
from refuel_oracle.schema import LLMAnnotation

class BaseTask(ABC):

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def get_example_generation_template(self) -> PromptTemplate:
        pass

    @abstractmethod
    def get_prompt_template(self) -> PromptTemplate:
        pass

    @abstractmethod
    def construct_prompt(self, **kwargs) -> str:
        pass

    @abstractmethod
    def parse_llm_response(self, prompt: str, response: LLMResult) -> LLMAnnotation:
        pass

    
    

