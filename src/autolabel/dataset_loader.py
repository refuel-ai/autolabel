from typing import Dict, List, Tuple, Union

from loguru import logger
import pandas as pd
from sqlalchemy.sql.selectable import Selectable


from autolabel.configs import AutolabelConfig


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
        self.dataset = dataset
        self.config = config
        self.max_items = max_items
        self.start_index = start_index

        if isinstance(dataset, str):
            self.read_file(dataset, config, max_items, start_index)
        elif isinstance(dataset, pd.DataFrame):
            self.read_dataframe(dataset, config, start_index, max_items)

    def read_csv(
        self,
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

    def read_dataframe(
        self,
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

    def read_jsonl(
        self,
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

    def read_file(
        self,
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
            return self.read_csv(
                file, config, max_items=max_items, start_index=start_index
            )
        elif file.endswith(".jsonl"):
            return self.read_jsonl(
                file, config, max_items=max_items, start_index=start_index
            )
        else:
            raise ValueError(f"Unsupported file format: {file}")
