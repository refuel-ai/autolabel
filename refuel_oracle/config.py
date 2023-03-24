import json
from typing import Dict

from loguru import logger


class Config:
    def __init__(self, config_dict: Dict) -> None:
        self._validate()
        self.config = config_dict

    def _validate(self):
        # TODO: validate provider and model names, task, prompt and seed sets, etc
        return True

    def get(self, key: str):
        return self.config.get(key, None)

    @classmethod
    def from_json(cls, json_file_path: str):
        try:
            config_dict = json.load(open(json_file_path))
        except ValueError:
            logger.error("JSON file: {} not loaded successfully", json_file_path)
            return None

        return Config(config_dict)
