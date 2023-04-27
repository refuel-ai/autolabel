from typing import Dict, List, Any
from loguru import logger
import json


class DatasetConfig:
    INPUT_COLUMNS_KEY = "input_columns"
    LABEL_COLUMN_KEY = "label_column"
    SEED_EXAMPLES_KEY = "seed_examples"
    DATASET_SCHEMA_KEY = "dataset_schema"
    DELIMITER_KEY = "delimiter"
    LABELS_LIST_KEY = "labels_list"
    EMPTY_RESPONSE_KEY = "empty_response"

    DEFAULT_SEPARATOR = ","

    def __init__(self, config_dict: Dict) -> None:
        self.dict = config_dict

    def get(self, key: str, default_value: Any = None) -> Any:
        return self.dict.get(key, default_value)

    def keys(self) -> List:
        return list(self.dict.keys())

    def __getitem__(self, key):
        return self.dict[key]

    def get_input_columns(self) -> List:
        return self.dict[self.DATASET_SCHEMA_KEY][self.INPUT_COLUMNS_KEY]

    def get_label_column(self) -> str:
        return self.dict[self.DATASET_SCHEMA_KEY][self.LABEL_COLUMN_KEY]

    def get_labels_list(self) -> str:
        return self.dict[self.LABELS_LIST_KEY]

    def get_seed_examples(self) -> List:
        return self.dict[self.SEED_EXAMPLES_KEY]

    def get_delimiter(self) -> str:
        return self.dict[self.DATASET_SCHEMA_KEY].get("delimiter", ",")

    @classmethod
    def from_json(cls, json_file_path: str, **kwargs):
        try:
            config_dict = json.load(open(json_file_path))
        except ValueError:
            logger.error("JSON file: {} not loaded successfully", json_file_path)
            return None

        config_dict.update(kwargs)

        return DatasetConfig(config_dict)
