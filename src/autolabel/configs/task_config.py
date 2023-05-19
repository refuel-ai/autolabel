from typing import Dict, Union

from .base import BaseConfig


class TaskConfig(BaseConfig):
    """Sets the task configuration of Autolabel.

    Attributes:
        task_name (str): Name of the task.
        task_type (str): Type of the task.
        prefix_prompt (str): Prefix prompt for the task.
        task_prompt (str): Task prompt for the task.
        output_prompt (str): Output prompt for the task.
        example_prompt (str): Example prompt for the task.
        prompt_template (str): Prompt template for the task.
        output_format (str): Output format for the task.
        example_selector (Dict): Example selector for the task. Consists of two keys: "strategy" - one of fixed_few_shot, semantic_similarity, max_marginal_relevance; "num_examples" - number of examples to be selected.
        compute_confidence (bool): Whether or not to compute confidence score for the task.
        chain_of_thought (bool): Whether or not to use chain of thought for the task.
    """

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
        return self.config.get(self.COMPUTE_CONFIDENCE_KEY, False)

    def use_chain_of_thought(self) -> bool:
        return self.config.get(self.CHAIN_OF_THOUGHT_KEY)

    def get_empty_response(self) -> str:
        return self.config.get(self.EMPTY_RESPONSE_KEY, "")
