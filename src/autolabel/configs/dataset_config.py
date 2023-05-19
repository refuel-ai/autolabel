from typing import Dict, List, Union

from .base import BaseConfig


class DatasetConfig(BaseConfig):
    """Sets the dataset configuration of Autolabel.

    Attributes:
        input_columns (List): List of input columns.
        label_column (str): Label column.
        seed_examples (Optional[str, List[Dict]]): Pass in the list of seed examples as the path to a csv or pass in the list of seed examples directly as a list of dictionaries.
        delimiter (str): Delimiter for the dataset.
    """
    INPUT_COLUMNS_KEY = "input_columns"
    LABEL_COLUMN_KEY = "label_column"
    SEED_EXAMPLES_KEY = "seed_examples"
    DATASET_SCHEMA_KEY = "dataset_schema"
    DELIMITER_KEY = "delimiter"
    LABELS_LIST_KEY = "labels_list"
    EMPTY_RESPONSE_KEY = "empty_response"

    DEFAULT_SEPARATOR = ","

    def __init__(self, config: Union[str, Dict]) -> None:
        super().__init__(config)

    def get_input_columns(self) -> List:
        return self.config[self.DATASET_SCHEMA_KEY][self.INPUT_COLUMNS_KEY]

    def get_label_column(self) -> str:
        return self.config[self.DATASET_SCHEMA_KEY][self.LABEL_COLUMN_KEY]

    def get_labels_list(self) -> str:
        return self.config[self.LABELS_LIST_KEY]

    def get_seed_examples(self) -> List:
        return self.config.get(self.SEED_EXAMPLES_KEY, [])

    def get_delimiter(self) -> str:
        return self.config[self.DATASET_SCHEMA_KEY].get(
            self.DELIMITER_KEY, self.DEFAULT_SEPARATOR
        )
