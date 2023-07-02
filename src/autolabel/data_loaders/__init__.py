import logging
import pandas as pd
from typing import Dict, Union
from tabulate import tabulate
from datasets import Dataset
from autolabel.data_loaders.read_datasets import (
    DataAttribute,
    CSVReader,
    JsonlReader,
    HuggingFaceDataset,
    # SqlDataset,
    DataFrameDataset,
)
from autolabel.data_loaders.validation import TaskDataValidation
from autolabel.configs import AutolabelConfig

logger = logging.getLogger(__name__)


class DataValidationFailed(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class DatasetLoader:
    # TODO: add support for reading from SQL databases
    # TODO: add support for reading and loading datasets in chunks
    MAX_ERROR_DISPLAYED = 100

    def __init__(
        self,
        dataset: Union[str, pd.DataFrame],
        config: AutolabelConfig,
        max_items: int = 0,
        start_index: int = 0,
    ) -> None:
        """DatasetLoader class to read and load datasets.

        Args:
            dataset (Union[str, pd.DataFrame]): path to the dataset or the dataframe
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to 0.
            start_index (int, optional): start index to read from. Defaults to 0.
        """
        self.dataset = dataset
        self.config = config
        self.max_items = max_items
        self.start_index = start_index

        self.__data_attr: DataAttribute = None
        self.__malformed_records = None
        self._read()

    @property
    def dat(
        self,
    ) -> Union[pd.DataFrame, Dataset]:
        return self.__data_attr.dataset

    @property
    def inputs(
        self,
    ) -> Dict:
        return self.__data_attr.inputs

    @property
    def gt_labels(
        self,
    ) -> str:
        return self.__data_attr.gt_labels

    @property
    def columns(
        self,
    ) -> str:
        return self.__data_attr.columns

    def _read(
        self,
    ):
        if isinstance(self.dataset, str):
            self.__data_attr: DataAttribute = self._read_file(
                self.dataset, self.config, self.max_items, self.start_index
            )
        elif isinstance(self.dataset, Dataset):
            self.__data_attr: DataAttribute = HuggingFaceDataset.read(
                self.dataset, self.config, self.max_items, self.start_index
            )
        elif isinstance(self.dataset, pd.DataFrame):
            self.__data_attr: DataAttribute = DataFrameDataset.read(
                self.dataset, self.config, self.start_index, self.max_items
            )

    def _read_file(
        self,
        file: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> None:
        """Read the file and sets dat, inputs and gt_labels

        Args:
            file (str): path to the file
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.

        Raises:
            ValueError: if the file format is not supported
        """
        if file.endswith(".csv"):
            return CSVReader.read(
                file, config, max_items=max_items, start_index=start_index
            )
        elif file.endswith(".jsonl"):
            return JsonlReader.read(
                file, config, max_items=max_items, start_index=start_index
            )
        else:
            raise ValueError(f"Unsupported file format: {file}")

    def validate(self):
        """Validate Data"""
        data_validation = TaskDataValidation(
            task_type=self.config.task_type(),
            label_column=self.config.label_column(),
            example_template=self.config.example_template(),
        )

        # Validate columns
        data_validation.validate_dataset_columns(
            dataset_columns=self.__data_attr.columns
        )

        # Validate datatype and data format
        self.__malformed_records = data_validation.validate(
            data=self.__data_attr.inputs
        )

        table = tabulate(
            self.__malformed_records[0 : self.MAX_ERROR_DISPLAYED],
            headers="keys",
            tablefmt="fancy_grid",
            numalign="center",
            stralign="left",
        )

        print(table)
        if len(self.__malformed_records) > 0:
            raise DataValidationFailed(
                f"Validation failed for {len(self.__malformed_records)} rows."
            )
