from typing import Dict, List, Union, Optional

import logging
import pandas as pd

from pydantic import BaseModel, validator
from datasets import Dataset
from sqlalchemy.sql.selectable import Selectable
from autolabel.configs import AutolabelConfig

logger = logging.getLogger(__name__)


class AutolabelDataset(BaseModel):
    """Data Attributes"""

    columns: List
    dataset: Union[pd.DataFrame, None]
    inputs: List[Dict]
    gt_labels: Optional[List]

    @validator("dataset", allow_reuse=True)
    def validate_dataframe(cls, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Value must be a pandas DataFrame")
        return value

    class Config:
        arbitrary_types_allowed = True


class CSVReader:
    @staticmethod
    def read(
        csv_file: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> AutolabelDataset:
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
        return AutolabelDataset(
            columns=list(dat.columns),
            dataset=dat,
            inputs=inputs,
            gt_labels=gt_labels,
        )


class JsonlReader:
    @staticmethod
    def read(
        jsonl_file: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> AutolabelDataset:
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
        return AutolabelDataset(
            columns=list(dat.columns),
            dataset=dat,
            inputs=inputs,
            gt_labels=gt_labels,
        )


class HuggingFaceDatasetReader:
    @staticmethod
    def read(
        dataset: Dataset,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> AutolabelDataset:
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
        return AutolabelDataset(
            columns=list(dat.columns),
            dataset=dat,
            inputs=inputs,
            gt_labels=gt_labels,
        )


class SqlDatasetReader:
    @staticmethod
    def read(
        sql: Union[str, Selectable],
        connection: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> AutolabelDataset:
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
        return AutolabelDataset(
            columns=list(dat.columns),
            dataset=dat,
            inputs=inputs,
            gt_labels=gt_labels,
        )


class DataframeReader:
    @staticmethod
    def read(
        df: pd.DataFrame,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> AutolabelDataset:
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
        return AutolabelDataset(
            columns=list(dat.columns),
            dataset=dat,
            inputs=inputs,
            gt_labels=gt_labels,
        )
