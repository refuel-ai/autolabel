from typing import Dict, List, Union

from .base import BaseConfig


class DatasetConfig(BaseConfig):
    INPUT_COLUMNS_KEY = "input_columns"
    EXAMPLE_TEMPLATE_KEY = "example_template"
    LABEL_COLUMN_KEY = "label_column"
    EXPLANATION_COLUMN_KEY = "explanation_column"
    SEED_EXAMPLES_KEY = "seed_examples"
    DATASET_SCHEMA_KEY = "dataset_schema"
    DELIMITER_KEY = "delimiter"
    LABELS_LIST_KEY = "labels_list"
    EMPTY_RESPONSE_KEY = "empty_response"

    DEFAULT_SEPARATOR = ","

    def __init__(self, config: Union[str, Dict]) -> None:
        super().__init__(config)

    def get_input_columns(self) -> List:
        """
        Returns a list of column names that will be used as input for annotation
        """
        return self.config[self.DATASET_SCHEMA_KEY][self.INPUT_COLUMNS_KEY]

    def get_example_template(self) -> str:
        """
        Returns a list of column names that will be used as input for annotation
        """
        return self.config.get(self.EXAMPLE_TEMPLATE_KEY, None)

    def get_label_column(self) -> str:
        """
        Returns the name of the column containing labels for dataset
        """
        return self.config[self.DATASET_SCHEMA_KEY].get(self.LABEL_COLUMN_KEY, None)
    
    def get_explanation_column(self) -> str:
        """
        Returns the name of the column containing explanations for dataset
        """
        return self.config[self.DATASET_SCHEMA_KEY].get(self.EXPLANATION_COLUMN_KEY, None)

    def get_labels_list(self) -> str:
        """
        Returns a list of valid labels for dataset
        """
        return self.config[self.LABELS_LIST_KEY]

    def get_seed_examples(self) -> List:
        """
        Returns a list of example datapoints with labels, used to prompt the LLM
        """
        return self.config[self.SEED_EXAMPLES_KEY]

    def get_delimiter(self) -> str:
        """
        Returns the string used to seperate values in dataset. Defaults to ','
        """
        return self.config[self.DATASET_SCHEMA_KEY].get(
            self.DELIMITER_KEY, self.DEFAULT_SEPARATOR
        )
