
from typing import Dict, Union

from .base import BaseConfig

class TaskConfig(BaseConfig):
    # Standardized (and required) keys in the TaskConfig
    TASK_NAME_KEY = "task_name"
    TASK_TYPE_KEY = "task_type"
    PREFIX_PROMPT_KEY = "prefix_prompt"
    TASK_PROMPT_KEY = "task_prompt"
    OUTPUT_PROMPT_KEY = "output_prompt"
    EXAMPLE_PROMPT_TEMPLATE_KEY = "example_prompt"
    PROMPT_TEMPLATE_KEY = "prompt_template"
    OUTPUT_FORMAT_KEY = "output_format"
    EXAMPLE_SELECTOR_KEY = "example_selector"
    HAS_LOGPROB_KEY = "has_logprob"
    COMPUTE_CONFIDENCE_KEY = "compute_confidence"
    CHAIN_OF_THOUGHT_KEY = "chain_of_thought"
    EMPTY_RESPONSE_KEY = "empty_response"

    def __init__(self, config: Union[str, Dict]) -> None:
        super().__init__(config)

    def get_task_name(self) -> str:
        return self.config.get(self.TASK_NAME_KEY, "new_task")

    def get_task_type(self) -> str:
        return self.config[self.TASK_TYPE_KEY]

    def get_prefix_prompt(self) -> str:
        return self.config.get(self.PREFIX_PROMPT_KEY, "")

    def get_task_prompt(self) -> str:
        return self.config.get(self.TASK_PROMPT_KEY, "")

    def get_output_prompt(self) -> str:
        return self.config.get(self.OUTPUT_PROMPT_KEY, "")

    def get_example_prompt_template(self) -> str:
        return self.config.get(self.EXAMPLE_PROMPT_TEMPLATE_KEY, "")

    def get_prompt_template(self) -> str:
        return self.config.get(self.PROMPT_TEMPLATE_KEY, "")

    def get_output_format(self) -> str:
        return self.config.get(self.OUTPUT_FORMAT_KEY, "json")

    def get_example_selector(self) -> str:
        return self.config.get(self.EXAMPLE_SELECTOR_KEY, {})

    def get_compute_confidence(self) -> bool:
        return self.config.get(self.COMPUTE_CONFIDENCE_KEY, "False") == "True"

    def use_chain_of_thought(self) -> bool:
        return self.config.get(self.CHAIN_OF_THOUGHT_KEY, "False") == "True"

    def get_empty_response(self) -> str:
        return self.config.get(self.EMPTY_RESPONSE_KEY, "")

