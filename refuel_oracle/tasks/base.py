"""Base interface that all prediction tasks will implement."""

from abc import ABC, abstractmethod
from typing import Dict, List

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Generation
from refuel_oracle.config import Config
from refuel_oracle.schema import LLMAnnotation, MetricResult


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
    def parse_llm_response(self, response: Generation, input: str) -> LLMAnnotation:
        pass

    @abstractmethod
    def eval(self, llm_labels: List, gt_labels: List) -> List[MetricResult]:
        pass

    def get_single_input(self, input: Dict) -> str:
        dataset_schema = self.config.get("dataset_schema", {})
        if not dataset_schema:
            raise ValueError("Dataset schema not found in config")

        if dataset_schema.get("input_template", ""):
            current_input = dataset_schema["input_template"].format(**input)
        else:
            input_column_list = dataset_schema.get("input_columns", [])
            if len(input_column_list) != 1:
                raise ValueError("Expected exactly one input column in dataset schema")

            current_input = input[input_column_list[0]]

        return current_input
