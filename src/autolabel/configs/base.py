import json
from typing import Any, Dict, List, Union

import logging

logger = logging.getLogger(__name__)


class BaseConfig:
    """Used for parsing, validating, and storing information about the labeling task passed to the LabelingAgent. Additional config classes should extend from this base class."""

    def __init__(self, config: Union[str, Dict]) -> None:
        if isinstance(config, str):
            self.config = self._safe_load_json(config)
        else:
            self.config = config
        self._validate()

    def _safe_load_json(self, json_file_path: str) -> Dict:
        """Loads config settings from a provided json file"""
        try:
            with open(json_file_path, "r") as config_file:
                return json.load(config_file)
        except ValueError as e:
            logger.error(
                f"JSON file: {json_file_path} not loaded successfully. Error: {repr(e)}"
            )
            return {}

    def _validate(self) -> bool:
        """Returns true if the config settings are valid"""
        return True

    def get(self, key: str, default_value: Any = None) -> Any:
        return self.config.get(key, default_value)

    def keys(self) -> List:
        return list(self.config.keys())

    def __getitem__(self, key):
        return self.config[key]

    def to_json(self) -> str:
        """Returns the BaseConfig object in JSON format"""
        return json.dumps(self.config, sort_keys=True)

    def __str__(self):
        return self.to_json()
