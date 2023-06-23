from typing import Dict, List, Tuple, Union

import logging
import pandas as pd
from sqlalchemy.sql.selectable import Selectable
from datasets import Dataset


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

        if isinstance(dataset, str):
            self._read_file(dataset, config, max_items, start_index)
        elif isinstance(dataset, Dataset):
            self._read_hf_dataset(dataset, config, max_items, start_index)
        elif isinstance(dataset, pd.DataFrame):
            self._read_dataframe(dataset, config, start_index, max_items)

    def _read_csv(
        self,
        csv_file: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> None:
        """Read the csv file and sets dat, inputs and gt_labels

        Args:
            csv_file (str): path to the csv file
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.
        """
        logger.debug(f"reading the csv from: {start_index}")
        delimiter = config.delimiter()
        label_column = config.label_column()

        self.dat = pd.read_csv(csv_file, sep=delimiter, dtype="str")[start_index:]
        self.dat = self.dat.astype(str)
        if max_items and max_items > 0:
            max_items = min(max_items, len(self.dat))
            self.dat = self.dat[:max_items]

        self.inputs = self.dat.to_dict(orient="records")
        self.gt_labels = (
            None
            if not label_column
            or not len(self.inputs)
            or label_column not in self.inputs[0]
            else self.dat[label_column].tolist()
        )

    def _read_dataframe(
        self,
        df: pd.DataFrame,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> None:
        """Read the csv file and sets dat, inputs and gt_labels

        Args:
            df (pd.DataFrame): dataframe to read
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.
        """
        label_column = config.label_column()

        self.dat = df[start_index:].astype(str)
        if max_items and max_items > 0:
            max_items = min(max_items, len(self.dat))
            self.dat = self.dat[:max_items]

        self.inputs = self.dat.to_dict(orient="records")
        self.gt_labels = (
            None
            if not label_column
            or not len(self.inputs)
            or label_column not in self.inputs[0]
            else self.dat[label_column].tolist()
        )

    def _read_jsonl(
        self,
        jsonl_file: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> None:
        """Read the jsonl file and sets dat, inputs and gt_labels

        Args:
            jsonl_file (str): path to the jsonl file
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.
        """
        logger.debug(f"reading the jsonl from: {start_index}")
        label_column = config.label_column()

        self.dat = pd.read_json(jsonl_file, lines=True, dtype="str")[start_index:]
        self.dat = self.dat.astype(str)
        if max_items and max_items > 0:
            max_items = min(max_items, len(self.dat))
            self.dat = self.dat[:max_items]

        self.inputs = self.dat.to_dict(orient="records")
        self.gt_labels = (
            None
            if not label_column
            or not len(self.inputs)
            or label_column not in self.inputs[0]
            else self.dat[label_column].tolist()
        )

    def _read_sql(
        self,
        sql: Union[str, Selectable],
        connection: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> None:
        """Read the sql query and sets dat, inputs and gt_labels

        Args:
            connection (str): connection string
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.
        """
        logger.debug(f"reading the sql from: {start_index}")
        label_column = config.label_column()

        self.dat = pd.read_sql(sql, connection)[start_index:]
        self.dat = self.dat.astype(str)
        if max_items and max_items > 0:
            max_items = min(max_items, len(self.dat))
            self.dat = self.dat[:max_items]

        self.inputs = self.dat.to_dict(orient="records")
        self.gt_labels = (
            None
            if not label_column
            or not len(self.inputs)
            or label_column not in self.inputs[0]
            else self.dat[label_column].tolist()
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
            return self._read_csv(
                file, config, max_items=max_items, start_index=start_index
            )
        elif file.endswith(".jsonl"):
            return self._read_jsonl(
                file, config, max_items=max_items, start_index=start_index
            )
        else:
            raise ValueError(f"Unsupported file format: {file}")

    def _read_hf_dataset(
        self,
        dataset: Dataset,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> None:
        """Read the huggingface dataset and sets dat, inputs and gt_labels

        Args:
            dataset (Dataset): dataset object to read from
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.
        """
        dataset.set_format("pandas")
        self.dat = dataset[
            start_index : max_items if max_items and max_items > 0 else len(dataset)
        ]

        self.inputs = self.dat.to_dict(orient="records")
        self.gt_labels = (
            None
            if not config.label_column()
            or not len(self.inputs)
            or config.label_column() not in self.inputs[0]
            else self.dat[config.label_column()].tolist()
        )
