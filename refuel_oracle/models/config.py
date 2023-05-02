import json
from typing import Any, Dict, List

from loguru import logger


class ModelConfig:
    # config keys
    LLM_PROVIDER_KEY = "provider_name"
    LLM_MODEL_KEY = "model_name"
    MODEL_PARAMS_KEY = "model_params"
    HAS_LOGPROB_KEY = "has_logprob"

    # default values
    DEFAULT_LLM_PROVIDER = "openai"

    def __init__(self, config_dict: Dict) -> None:
        self.dict = config_dict or {}

    def get(self, key: str, default_value: Any = None) -> Any:
        return self.dict.get(key, default_value)

    def keys(self) -> List:
        return list(self.dict.keys())

    def __getitem__(self, key):
        return self.dict[key]

    def get_provider(self) -> str:
        provider_name = self.dict.get(self.LLM_PROVIDER_KEY, self.DEFAULT_LLM_PROVIDER)
        return provider_name

    def get_model_name(self) -> str:
        return self.dict.get(self.LLM_MODEL_KEY, None)

    def get_model_params(self) -> Dict:
        return self.dict.get(self.MODEL_PARAMS_KEY, {})

    def get_has_logprob(self) -> bool:
        """
        Returns whether or not current task supports returning LogProb confidence of its response
        """
        return self.dict.get(self.HAS_LOGPROB_KEY, False)

    @classmethod
    def from_json(cls, json_file_path: str):
        try:
            with open(json_file_path, "r") as config_file:
                config_dict = json.load(config_file)
        except ValueError:
            logger.error("JSON file: {} not loaded successfully", json_file_path)
            return {}
        return ModelConfig(config_dict)
