from typing import Dict, List, Union

from .base import BaseConfig


class DatasetConfig(BaseConfig):
    """Sets the dataset configuration of Autolabel.

    Attributes:
        label_column (str): Label column.
        seed_examples (Optional[str, List[Dict]]): Pass in the list of seed examples as the path to a csv or pass in the list of seed examples directly as a list of dictionaries.
        delimiter (str): Delimiter for the dataset.
    """

    LABEL_COLUMN_KEY = "label_column"
    EXPLANATION_COLUMN_KEY = "explanation_column"
    SEED_EXAMPLES_KEY = "seed_examples"
    DELIMITER_KEY = "delimiter"
    LABELS_LIST_KEY = "labels_list"
    EMPTY_RESPONSE_KEY = "empty_response"
    EXAMPLE_TEMPLATE_KEY = "example_template"

    DEFAULT_SEPARATOR = ","

    def __init__(self, config: Union[str, Dict]) -> None:
        super().__init__(config)

    def get_label_column(self) -> str:
        """
        Returns the name of the column containing labels for dataset
        """
        return self.config.get(self.LABEL_COLUMN_KEY, None)

    def get_explanation_column(self) -> str:
        """
        Returns the name of the column containing explanations for dataset
        """
        return self.config.get(self.EXPLANATION_COLUMN_KEY, None)

    def get_labels_list(self) -> str:
        return self.config[self.LABELS_LIST_KEY]

    def get_seed_examples(self) -> List:
        return self.config.get(self.SEED_EXAMPLES_KEY, [])

    def get_delimiter(self) -> str:
        """
        Returns the string used to seperate values in dataset. Defaults to ','
        """
        return self.config.get(self.DELIMITER_KEY, self.DEFAULT_SEPARATOR)

    def get_example_template(self) -> str:
        """
        Returns a string to format an example input
        """
        return self.config.get(self.EXAMPLE_TEMPLATE_KEY, None)
