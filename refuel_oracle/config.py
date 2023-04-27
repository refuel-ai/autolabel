from ast import List
import json
from typing import Any, Dict

from loguru import logger


class Config:
    # Standardized (and required) keys in the Config
    PROJECT_NAME_KEY = "project_name"
    TASK_TYPE_KEY = "task_type"
    LLM_PROVIDER_KEY = "provider_name"
    LLM_MODEL_KEY = "model_name"

    def __init__(self, config_dict: Dict) -> None:
        self._validate()
        self.config = config_dict

    def _validate(self) -> bool:
        """
        validate provider and model names, task, prompt and seed sets, etc

        Returns:
            True if valid Config settings, False otherwise
        """
        # TODO: validate provider and model names, task, prompt and seed sets, etc
        return True

    def get(self, key: str, default_value: Any = None) -> Any:
        """
        Allow for dictionary like access of class members
        """
        return self.config.get(key, default_value)

    def keys(self) -> List:
        """
        Returns a list of keys stored within config object, similar to python dictionaries
        """
        return list(self.config.keys())

    def __getitem__(self, key):
        return self.config[key]

    def get_provider(self) -> str:
        """
        Returns the name of the provider (i.e. OpenAI, Anthropic, Huggingface) currently being used
        """
        return self.config[self.LLM_PROVIDER_KEY]

    def get_model_name(self) -> str:
        """
        Returns the name of the language model currently being used for annotation
        """
        return self.config[self.LLM_MODEL_KEY]

    def get_project_name(self) -> str:
        """
        Returns the name of the project, as defined in the json configuration file
        """
        return self.config[self.PROJECT_NAME_KEY]

    def get_task_type(self) -> str:
        """
        Returns the task that oracle is currently set to perform (i.e. classification, question answering, entity detection)
        """
        return self.config[self.TASK_TYPE_KEY]

    @classmethod
    def from_json(cls, json_file_path: str, **kwargs) -> Config:
        """
        parses a given config.json file and returns it in a new Config object

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

        return Config(config_dict)
