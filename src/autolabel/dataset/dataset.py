import logging
from typing import Callable, Dict, List, Union, Optional

import pandas as pd
from rich.console import Console
from tabulate import tabulate

from autolabel.configs import AutolabelConfig
from autolabel.dataset.validation import TaskDataValidation
from autolabel.schema import LLMAnnotation, MetricResult
from autolabel.tasks import BaseTask, TaskFactory
from autolabel.utils import print_table

logger = logging.getLogger(__name__)

METRIC_TABLE_STYLE = "cyan bold"


class AutolabelDataset:
    """The dataset for handling all operations on the dataset."""

    inputs: List[Dict]
    df: pd.DataFrame
    gt_labels: List
    config: AutolabelConfig

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        dataset: Union[pd.DataFrame, str],
        config: Union[AutolabelConfig, str, Dict],
        max_items: int = None,
        start_index: int = 0,
        validate: bool = False,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initializes the dataset.
        Args:
            dataset: The dataset to be used for labeling. Could be a path to a csv/jsonl file or a pandas dataframe.
            config: The config to be used for labeling. Could be a path to a json file or a dictionary.
            max_items: The maximum number of items to be parsed into the dataset object.
            start_index: The index to start parsing the dataset from.
            validate: Whether to validate the dataset or not.
        """
        if not (isinstance(config, AutolabelConfig)):
            self.config = AutolabelConfig(config)
        else:
            self.config = config

        if isinstance(dataset, str):
            if dataset.endswith(".csv"):
                delimiter = self.config.delimiter()
                quoting = 0
                if self.config.disable_quoting():
                    quoting = 3
                df = pd.read_csv(dataset, sep=delimiter, dtype="str", quoting=quoting)
            elif dataset.endswith(".jsonl"):
                df = pd.read_json(dataset, lines=True, dtype="str")
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        if shuffle:
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        df = df[start_index:]
        if max_items and max_items > 0:
            max_items = min(max_items, len(df))
            df = df[:max_items]

        inputs = df.to_dict(orient="records")
        label_column = self.config.label_column()
        gt_labels = (
            None
            if not label_column or not len(inputs) or label_column not in inputs[0]
            else df[label_column].tolist()
        )

        self.df = df
        self.inputs = inputs
        self.gt_labels = gt_labels

        if validate:
            self._validate()

    def __repr__(self):
        """
        Returns the representation of the dataset. We currently represent the dataset as a pandas dataframe.
        """
        if self.df is not None:
            return self.df.__repr__()

    def __str__(self):
        if self.df is not None:
            return self.df.__str__()

    def get_slice(self, max_items: int = None, start_index: int = 0):
        df = self.df[start_index:]
        if max_items and max_items > 0:
            max_items = min(max_items, len(df))
            df = df[:max_items]

        return AutolabelDataset(df, self.config)

    def process_labels(
        self, llm_labels: List[LLMAnnotation], metrics: List[MetricResult] = None
    ):
        # Add the LLM labels to the dataframe
        self.df[self.generate_label_name("label")] = [x.label for x in llm_labels]

        for attr in self.config.attributes():
            attribute_labels = []
            for x in llm_labels:
                if x.successfully_labeled:
                    attribute_labels.append(x.label.get(attr["name"], ""))
                else:
                    attribute_labels.append(BaseTask.NULL_LABEL_TOKEN)
            self.df[self.generate_label_name("label", attr["name"])] = attribute_labels

        # Add the LLM errors to the dataframe
        self.df[self.generate_label_name("error")] = [x.error for x in llm_labels]

        # Add the LLM prompts to the dataframe
        self.df[self.generate_label_name("prompt")] = [x.prompt for x in llm_labels]

        # Add labeled success column to the dataframe
        self.df[self.generate_label_name("successfully_labeled")] = [
            x.successfully_labeled for x in llm_labels
        ]

        # Add the LLM annotations to the dataframe
        self.df[self.generate_label_name("annotation")] = llm_labels

        # Add row level LLM metrics to the dataframe
        if metrics is not None:
            for metric in metrics:
                if (
                    isinstance(metric.value, list)
                    and len(metric.value) == self.df.shape[0]
                ):
                    self.df[self.generate_label_name(metric.name)] = metric.value

        # Add the LLM confidence scores to the dataframe if confidence is set in config
        if self.config.confidence():
            self.df[self.generate_label_name("confidence")] = [
                x.confidence_score for x in llm_labels
            ]
            for attr in self.config.attributes():
                attr_confidence_scores = []
                for x in llm_labels:
                    if x.successfully_labeled:
                        attr_confidence_scores.append(
                            x.confidence_score.get(attr["name"], 0.0)
                        )
                    else:
                        attr_confidence_scores.append(0.0)
                self.df[self.generate_label_name("confidence", attr["name"])] = (
                    attr_confidence_scores
                )

        # Add the LLM explanations to the dataframe if chain of thought is set in config
        if self.config.chain_of_thought():
            self.df[self.generate_label_name("explanation")] = [
                l.explanation for l in llm_labels
            ]

    def save(self, output_file_name: str):
        """
        Saves the dataset to a file based on the file extension.
        Args:
            output_file_name: The name of the file to save the dataset to. Based on the extension we can save to a csv or jsonl file.
        """
        if output_file_name.endswith(".csv"):
            self.df.to_csv(
                str(output_file_name),
                sep=self.config.delimiter(),
                header=True,
                index=False,
            )
        elif output_file_name.endswith(".jsonl"):
            self.df.to_json(
                str(output_file_name),
                orient="records",
                lines=True,
                force_ascii=False,
            )
        else:
            raise ValueError(f"Unsupported output file format: {output_file_name}")

    def filter(
        self,
        label: str = None,
        ground_truth: str = None,
        filter_func: Callable = None,
        label_column: str = None,
    ):
        """
        Filter the dataset based on the label, ground truth or a custom filter function.
        In case multiple filters are applied, the filters are applied in the following order:
            label -> ground_truth -> filter_func
        Args:
            label: The llm label to filter on.
            ground_truth: The ground truth label to filter on.
            filter_func: A custom filter function to filter on.
            label_column: The column to filter on. This is only used for attribute extraction tasks.
        """
        filtered_df = self.df

        if label:
            filtered_df = filtered_df[
                filtered_df[self.generate_label_name("label", label_column)] == label
            ]

        if ground_truth:
            filtered_df = filtered_df[
                filtered_df[(label_column or self.config.label_column())]
                == ground_truth
            ]

        if filter_func:
            filtered_df = filtered_df.apply(filter_func, axis=1)

        return AutolabelDataset(
            filtered_df,
            self.config,
        )

    def non_completed(self):
        """
        Filter the dataset to only include non completed items. This means the labels
        where the llm was not able to generate a label or there was some error while
        generating the label.
        """
        filtered_df = self.df[self.df[self.generate_label_name("error")].notnull()]
        return AutolabelDataset(filtered_df, self.config)

    def completed(self):
        """
        Filter the dataset to only include completed items. This means the labels
        where the llm was able to generate a label successfully.
        """
        filtered_df = self.df[self.df[self.generate_label_name("error")].isnull()]
        return AutolabelDataset(filtered_df, self.config)

    def incorrect(
        self, label: str = None, ground_truth: str = None, label_column: str = None
    ):
        """
        Filter the dataset to only include incorrect items. This means the labels
        where the llm label was incorrect.
        Args:
            label: The llm label to filter on.
            ground_truth: The ground truth label to filter on.
            label_column: The column to filter on. This is only used for attribute extraction tasks.
        """
        gt_label_column = label_column or self.config.label_column()

        if gt_label_column is None:
            raise ValueError(
                "Cannot compute mistakes without ground truth label column"
            )

        filtered_df = self.df[
            self.df[self.generate_label_name("label", label_column)]
            != self.df[gt_label_column]
        ]

        if label:
            filtered_df = filtered_df[
                filtered_df[self.generate_label_name("label", label_column)] == label
            ]

        if ground_truth:
            filtered_df = filtered_df[filtered_df[gt_label_column] == ground_truth]

        return AutolabelDataset(filtered_df, self.config)

    def correct(self, label_column: str = None):
        """
        Filter the dataset to only include correct items. This means the labels
        where the llm label was correct.
        Args:
            label_column: The column to filter on. This is only used for attribute extraction tasks.
        """
        gt_label_column = label_column or self.config.label_column()

        if gt_label_column is None:
            raise ValueError("Cannot compute correct without ground truth label column")

        filtered_df = self.df[
            self.df[self.generate_label_name("label", label_column)]
            == self.df[gt_label_column]
        ]
        return AutolabelDataset(filtered_df, self.config)

    def filter_by_confidence(self, threshold: float = 0.5):
        """
        Filter the dataset to only include items with confidence scores greater than the threshold.
        Args:
            threshold: The threshold to filter on. This means that only items with confidence scores greater than the threshold will be included.
        """
        if not self.config.confidence():
            raise ValueError(
                "Cannot compute correct and confident without confidence scores"
            )

        filtered_df = self.df[
            self.df[self.generate_label_name("confidence")] >= threshold
        ]
        return AutolabelDataset(filtered_df, self.config)

    def eval(self):
        """
        Evaluate the dataset based on the task. We run the metrics that were
        specified by the task being run.
        """
        llm_labels = self.df[self.generate_label_name("annotation")].tolist()

        task = TaskFactory.from_config(self.config)

        metrics = task.eval(llm_labels, self.gt_labels)

        table = {}
        for metric in metrics:
            if not isinstance(metric.value, list):
                table[metric.name] = metric.value

        print_table(table, console=Console(), default_style=METRIC_TABLE_STYLE)

        return metrics

    def columns(self):
        """
        Returns the columns in the dataframe.
        """
        return self.df.columns.tolist()

    def _validate(self):
        """
        Validate the dataset by looking at all rows and making sure
        that they follow the schema.
        """
        data_validation = TaskDataValidation(config=self.config)

        # Validate columns
        data_validation.validate_dataset_columns(dataset_columns=self.columns())

        # Validate datatype and data format
        self.__malformed_records = data_validation.validate(data=self.inputs)

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

    def generate_label_name(self, col_name: str, label_column: str = None):
        label_column = label_column or f"{self.config.task_name()}_task"
        return f"{label_column}_{col_name}"


class DataValidationFailed(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
