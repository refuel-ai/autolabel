from typing import Dict, List, Tuple, Union

import logging
import pandas as pd
from sqlalchemy.sql.selectable import Selectable


from autolabel.configs import AutolabelConfig

logger = logging.getLogger(__name__)


class DatasetLoader:
    # TODO: add support for reading from SQL databases
    # TODO: add support for reading and loading datasets in chunks

    @staticmethod
    def read_csv(
        csv_file: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> Tuple[pd.DataFrame, List[Dict], List]:
        """Read the csv file and return the dataframe, inputs and gt_labels

        Args:
            csv_file (str): path to the csv file
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.

        Returns:
            Tuple[pd.DataFrame, List[Dict], List]: dataframe, inputs and gt_labels
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
        return (dat, inputs, gt_labels)

    @staticmethod
    def read_dataframe(
        df: pd.DataFrame,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> Tuple[pd.DataFrame, List[Dict], List]:
        """Read the csv file and return the dataframe, inputs and gt_labels

        Args:
            df (pd.DataFrame): dataframe to read
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.

        Returns:
            Tuple[pd.DataFrame, List[Dict], List]: dataframe, inputs and gt_labels
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
        return (dat, inputs, gt_labels)

    @staticmethod
    def read_jsonl(
        jsonl_file: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> Tuple[pd.DataFrame, List[Dict], List]:
        """Read the jsonl file and return the dataframe, inputs and gt_labels

        Args:
            jsonl_file (str): path to the jsonl file
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.

        Returns:
            Tuple[pd.DataFrame, List[Dict], List]: dataframe, inputs and gt_labels
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
        return (dat, inputs, gt_labels)

    @staticmethod
    def read_sql(
        self,
        sql: Union[str, Selectable],
        connection: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> Tuple[pd.DataFrame, List[Dict], List]:
        """Read the sql query and return the dataframe, inputs and gt_labels

        Args:
            connection (str): connection string
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.

        Returns:
            Tuple[pd.DataFrame, List[Dict], List]: dataframe, inputs and gt_labels
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
        return (dat, inputs, gt_labels)

    @staticmethod
    def read_file(
        file: str,
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
    ) -> Tuple[pd.DataFrame, List[Dict], List]:
        """Read the file and return the dataframe, inputs and gt_labels

        Args:
            file (str): path to the file
            config (AutolabelConfig): config object
            max_items (int, optional): max number of items to read. Defaults to None.
            start_index (int, optional): start index to read from. Defaults to 0.

        Raises:
            ValueError: if the file format is not supported

        Returns:
            Tuple[pd.DataFrame, List[Dict], List]: dataframe, inputs and gt_labels
        """
        if file.endswith(".csv"):
            return DatasetLoader.read_csv(
                file, config, max_items=max_items, start_index=start_index
            )
        elif file.endswith(".jsonl"):
            return DatasetLoader.read_jsonl(
                file, config, max_items=max_items, start_index=start_index
            )
        else:
            raise ValueError(f"Unsupported file format: {file}")
