

"""Base interface that all prediction tasks will implement."""

from abc import ABC, abstractmethod
from typing import List
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation

from refuel_oracle.config import Config
from refuel_oracle.schema import LLMAnnotation

class BaseTask(ABC):

    def __init__(self, config: Config) -> None:
        self.config = config
        self.prompt_template = self.initialize_prompt_template()

    @abstractmethod
    def initialize_prompt_template(self) -> PromptTemplate:
        pass

    @abstractmethod
    def construct_prompt(self, input: str, examples: List) -> str:
        pass

    @abstractmethod
    def parse_llm_response(self, prompt: str, response: Generation) -> LLMAnnotation:
        pass

    
    

