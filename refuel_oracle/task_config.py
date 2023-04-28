from ast import List
import json
from typing import Any, Dict

from loguru import logger


class TaskConfig:
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
    EMPTY_RESPONSE_KEY = "empty_response"

    def __init__(self, config_dict: Dict) -> None:
        self._validate()
        self.config = config_dict

    def _validate(self) -> bool:
        """
        Returns:
            True if valid TaskConfig settings, False otherwise
        """
        # TODO: validate provider and model names, task, prompt and seed sets, etc
        return True

    def get(self, key: str, default_value: Any = None) -> Any:
        return self.config.get(key, default_value)

    def keys(self) -> List:
        return list(self.config.keys())

    def __getitem__(self, key):
        return self.config[key]

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

    def get_has_logprob(self) -> str:
        """
        Returns whether or not current task supports returning LogProb confidence of its response
        """
        return self.config.get(self.HAS_LOGPROB_KEY, "False")

    @classmethod
    def from_json(cls, json_file_path: str, **kwargs):
        """
        parses a given json file for task settings and returns it in a new Config object

        Args:
            json_file_path: path to json configuration file
            **kwargs: additional settings not found in json file can be passed from here

        Returns:
            Config object containing project settings found in json_file_path
        """
        try:
            config_dict = json.load(open(json_file_path))
        except ValueError:
            logger.error("JSON file: {} not loaded successfully", json_file_path)
            return None

        config_dict.update(kwargs)

        return TaskConfig(config_dict)
