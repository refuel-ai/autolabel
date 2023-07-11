import logging
from typing import Dict, Union

import pandas as pd
from autolabel.configs import AutolabelConfig
from autolabel.data_loaders.read_datasets import (  # SqlDatasetReader,
    AutolabelDataset,
    CSVReader,
    DataframeReader,
    HuggingFaceDatasetReader,
    JsonlReader,
)
from autolabel.data_loaders.validation import TaskDataValidation
from datasets import Dataset
from tabulate import tabulate

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
        validate: bool = True,
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

        self.__al_dataset: AutolabelDataset = None
        self.__malformed_records = None
        self._read()
        if validate:
            self._validate()

    @property
    def dat(
        self,
    ) -> Union[pd.DataFrame, Dataset]:
        return self.__al_dataset.dataset

    @property
    def inputs(
        self,
    ) -> Dict:
        return self.__al_dataset.inputs

    @property
    def gt_labels(
        self,
    ) -> str:
        return self.__al_dataset.gt_labels

    @property
    def columns(
        self,
    ) -> str:
        return self.__al_dataset.columns

    def _read(
        self,
    ):
        if isinstance(self.dataset, str):
            if self.dataset.endswith(".csv"):
                self.__al_dataset = CSVReader.read(
                    self.dataset,
                    self.config,
                    max_items=self.max_items,
                    start_index=self.start_index,
                )

            elif self.dataset.endswith(".jsonl"):
                self.__al_dataset = JsonlReader.read(
                    self.dataset,
                    self.config,
                    max_items=self.max_items,
                    start_index=self.start_index,
                )
            else:
                raise ValueError(f"Unsupported file format: {self.dataset}")
        elif isinstance(self.dataset, Dataset):
            self.__al_dataset: AutolabelDataset = HuggingFaceDatasetReader.read(
                self.dataset, self.config, self.max_items, self.start_index
            )
        elif isinstance(self.dataset, pd.DataFrame):
            self.__al_dataset: AutolabelDataset = DataframeReader.read(
                self.dataset, self.config, self.start_index, self.max_items
            )

    def _validate(self):
        """Validate Data"""
        data_validation = TaskDataValidation(config=self.config)

        # Validate columns
        data_validation.validate_dataset_columns(
            dataset_columns=self.__al_dataset.columns
        )

        # Validate datatype and data format
        self.__malformed_records = data_validation.validate(
            data=self.__al_dataset.inputs
        )

        table = tabulate(
            self.__malformed_records[0 : self.MAX_ERROR_DISPLAYED],
            headers="keys",
            tablefmt="fancy_grid",
            numalign="center",
            stralign="left",
        )

        if len(self.__malformed_records) > 0:
            logger.warning(
                f"Data Validation failed for {len(self.__malformed_records)} records: \n Stats: \n {table}"
            )
            raise DataValidationFailed(
                f"Validation failed for {len(self.__malformed_records)} rows."
            )
