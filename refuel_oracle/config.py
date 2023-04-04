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
        # TODO: validate provider and model names, task, prompt and seed sets, etc
        return True

    def get(self, key: str, default_value: Any = None) -> Any:
        return self.config.get(key, default_value)

    def keys(self) -> List:
        return list(self.config.keys())

    def __getitem__(self, key):
        return self.config[key]

    def get_provider(self) -> str:
        return self.config[self.LLM_PROVIDER_KEY]

    def get_model_name(self) -> str:
        return self.config[self.LLM_MODEL_KEY]
    
    def get_project_name(self) -> str:
        return self.config[self.PROJECT_NAME_KEY]

    def get_task_type(self) -> str:
        return self.config[self.TASK_TYPE_KEY]

    @classmethod
    def from_json(cls, json_file_path: str):
        try:
            config_dict = json.load(open(json_file_path))
        except ValueError:
            logger.error("JSON file: {} not loaded successfully", json_file_path)
            return None

        return Config(config_dict)
