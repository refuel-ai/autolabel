from typing import Dict, Union

import logging
import pandas as pd
from datasets import Dataset
from autolabel.data_loaders.read_datasets import (
    DataLoaderAttribute,
    CSVReader,
    JsonlReader,
    HuggingFaceDataset,
    SqlDataset,
    DataFrameDataset,
)

from autolabel.configs import AutolabelConfig

logger = logging.getLogger(__name__)


class DatasetLoader:
    # TODO: add support for reading from SQL databases
    # TODO: add support for reading and loading datasets in chunks

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

        self.__data_attr: DataLoaderAttribute = None

        self._read()

    def _read(
        self,
    ):
        if isinstance(self.dataset, str):
            self.__data_attr: DataLoaderAttribute = self._read_file(
                self.dataset, self.config, self.max_items, self.start_index
            )
        elif isinstance(self.dataset, Dataset):
            self.__data_attr: DataLoaderAttribute = HuggingFaceDataset.read(
                self.dataset, self.config, self.max_items, self.start_index
            )
        elif isinstance(self.dataset, pd.DataFrame):
            self.__data_attr: DataLoaderAttribute = DataFrameDataset.read(
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
