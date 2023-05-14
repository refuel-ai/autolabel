import json
from typing import Any, Dict, List, Union

from loguru import logger


class BaseConfig:
    def __init__(self, config: Union[str, Dict]) -> None:
        if isinstance(config, str):
            self.config = self._safe_load_json(config)
        else:
            self.config = config

    def _safe_load_json(self, json_file_path: str) -> Dict:
        try:
            with open(json_file_path, "r") as config_file:
                return json.load(config_file)
        except ValueError as e:
            logger.error(
                f"JSON file: {json_file_path} not loaded successfully. Error: {repr(e)}"
            )
            return {}

    def get(self, key: str, default_value: Any = None) -> Any:
        return self.config.get(key, default_value)

    def keys(self) -> List:
        return list(self.config.keys())

    def __getitem__(self, key):
        return self.config[key]

    def to_json(self) -> str:
        return json.dumps(self.config, sort_keys=True)
