from typing import Dict, List, Tuple, Union

import logging
import pandas as pd
from dataclasses import dataclass
from datasets import Dataset
from sqlalchemy.sql.selectable import Selectable
from autolabel.configs import AutolabelConfig

logger = logging.getLogger(__name__)
from typing import Union


@dataclass
class DataLoaderAttribute:
    dataset: Union[pd.DataFrame, Dataset]
    inputs: Dict
    gt_labels: List


class CSVReader:
    @staticmethod
    def read(
        csv_file: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> DataLoaderAttribute:
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

        dat = pd.read_csv(csv_file, sep=delimiter, dtype="str")[start_index:]
        dat = dat.astype(str)
        if max_items and max_items > 0:
            max_items = min(max_items, len(dat))
            dat = dat[:max_items]

        inputs = dat.to_dict(orient="records")
        gt_labels = (
            None
            if not label_column or not len(inputs) or label_column not in inputs[0]
            else dat[label_column].tolist()
        )
        return DataLoaderAttribute(
            dat,
            inputs,
            gt_labels,
        )


class JsonlReader:
    @staticmethod
    def read(
        jsonl_file: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> DataLoaderAttribute:
        """Read the jsonl file and sets dat, inputs and gt_labels

        Args:
            jsonl_file (str): path to the jsonl file
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.
        """
        logger.debug(f"reading the jsonl from: {start_index}")
        label_column = config.label_column()

        dat = pd.read_json(jsonl_file, lines=True, dtype="str")[start_index:]
        dat = dat.astype(str)
        if max_items and max_items > 0:
            max_items = min(max_items, len(dat))
            dat = dat[:max_items]

        inputs = dat.to_dict(orient="records")
        gt_labels = (
            None
            if not label_column or not len(inputs) or label_column not in inputs[0]
            else dat[label_column].tolist()
        )
        return DataLoaderAttribute(
            dat,
            inputs,
            gt_labels,
        )


class HuggingFaceDataset:
    @staticmethod
    def read(
        dataset: Dataset,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> DataLoaderAttribute:
        """Read the huggingface dataset and sets dat, inputs and gt_labels

        Args:
            dataset (Dataset): dataset object to read from
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.
        """
        dataset.set_format("pandas")
        dat = dataset[
            start_index : max_items if max_items and max_items > 0 else len(dataset)
        ]

        inputs = dat.to_dict(orient="records")
        gt_labels = (
            None
            if not config.label_column()
            or not len(inputs)
            or config.label_column() not in inputs[0]
            else dat[config.label_column()].tolist()
        )
        return DataLoaderAttribute(dat, inputs, gt_labels)


class SqlDataset:
    @staticmethod
    def read(
        sql: Union[str, Selectable],
        connection: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> DataLoaderAttribute:
        """Read the sql query and sets dat, inputs and gt_labels

        Args:
            connection (str): connection string
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.
        """
        logger.debug(f"reading the sql from: {start_index}")
        label_column = config.label_column()

        dat = pd.read_sql(sql, connection)[start_index:]
        dat = dat.astype(str)
        if max_items and max_items > 0:
            max_items = min(max_items, len(dat))
            dat = dat[:max_items]

        inputs = dat.to_dict(orient="records")
        gt_labels = (
            None
            if not label_column or not len(inputs) or label_column not in inputs[0]
            else dat[label_column].tolist()
        )
        return DataLoaderAttribute(
            dat,
            inputs,
            gt_labels,
        )


class DataFrameDataset:
    @staticmethod
    def read(
        df: pd.DataFrame,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> DataLoaderAttribute:
        """Read the csv file and sets dat, inputs and gt_labels

        Args:
            df (pd.DataFrame): dataframe to read
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.
        """
        label_column = config.label_column()

        dat = df[start_index:].astype(str)
        if max_items and max_items > 0:
            max_items = min(max_items, len(dat))
            dat = dat[:max_items]

        inputs = dat.to_dict(orient="records")
        gt_labels = (
            None
            if not label_column or not len(inputs) or label_column not in inputs[0]
            else dat[label_column].tolist()
        )
        return DataLoaderAttribute(
            dat,
            inputs,
            gt_labels,
        )
