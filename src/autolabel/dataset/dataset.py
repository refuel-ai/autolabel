from typing import Dict, List, Union
import pandas as pd
from autolabel.configs import AutolabelConfig
from autolabel.dataset.validation import TaskDataValidation
from autolabel.schema import MetricResult, LLMAnnotation
from tabulate import tabulate
import logging
from autolabel.utils import print_table
from rich.console import Console
import json
from autolabel.tasks import TaskFactory

logger = logging.getLogger(__name__)

METRIC_TABLE_STYLE = "cyan bold"


class AutolabelDataset:
    """Data Attributes"""

    inputs: List[Dict]
    df: pd.DataFrame
    gt_labels: List
    config: AutolabelConfig

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        dataset: Union[pd.DataFrame, str],
        config: AutolabelConfig,
        max_items: int = None,
        start_index: int = 0,
        validate=False,
    ) -> None:
        if isinstance(dataset, str):
            if dataset.endswith(".csv"):
                delimiter = config.delimiter()
                df = pd.read_csv(dataset, sep=delimiter, dtype="str")
            elif dataset.endswith(".jsonl"):
                df = pd.read_json(dataset, lines=True, dtype="str")
        elif isinstance(dataset, pd.DataFrame):
            df = dataset

        df = df[start_index:]
        if max_items and max_items > 0:
            max_items = min(max_items, len(df))
            df = df[:max_items]

        inputs = df.to_dict(orient="records")
        label_column = config.label_column()
        gt_labels = (
            None
            if not label_column or not len(inputs) or label_column not in inputs[0]
            else df[label_column].tolist()
        )

        self.df = df
        self.inputs = inputs
        self.gt_labels = gt_labels
        self.config = config
        self.prefix = self.config.task_name() + "_"

        if validate:
            self._validate()

    def __repr__(self):
        if self.df is not None:
            return self.df.__repr__()

    def __str__(self):
        if self.df is not None:
            return self.df.__str__()

    def process_labels(
        self, llm_labels: List[LLMAnnotation], metrics: List[MetricResult] = None
    ):
        # Add the LLM labels to the dataframe
        self.df[self.prefix + "label"] = [x.label for x in llm_labels]

        # Add the LLM errors to the dataframe
        self.df[self.prefix + "error"] = [x.error for x in llm_labels]

        # Add labeled success column to the dataframe
        self.df[self.prefix + "successfully_labeled"] = [
            x.successfully_labeled for x in llm_labels
        ]

        self.df[self.prefix + "annotation"] = [x.json() for x in llm_labels]

        # Add row level LLM metrics to the dataframe
        if metrics is not None:
            for metric in metrics:
                if (
                    isinstance(metric.value, list)
                    and len(metric.value) == self.df.shape[0]
                ):
                    self.df[self.prefix + metric.name] = metric.value

        # Add the LLM confidence scores to the dataframe if confidence is set in config
        if self.config.confidence():
            self.df[self.prefix + "confidence"] = [
                x.confidence_score for x in llm_labels
            ]

        # Add the LLM explanations to the dataframe if chain of thought is set in config
        if self.config.chain_of_thought():
            self.df[self.prefix + "explanation"] = [l.explanation for l in llm_labels]

    def save(self, output_file_name: str):
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

    def filter(self, label=None, ground_truth=None, filter_func=None):
        filtered_df = self.df

        if label:
            filtered_df = filtered_df[filtered_df[self.prefix + "label"] == label]

        if ground_truth:
            filtered_df = filtered_df[
                filtered_df[self.config.label_column()] == ground_truth
            ]

        if filter_func:
            filtered_df = filtered_df.apply(filter_func, axis=1)

        return AutolabelDataset(
            filtered_df,
            self.config,
        )

    def errors(self):
        filtered_df = self.df[self.df[self.prefix + "error"].notnull()]
        return AutolabelDataset(filtered_df, self.config)

    def non_errors(self):
        filtered_df = self.df[self.df[self.prefix + "error"].isnull()]
        return AutolabelDataset(filtered_df, self.config)

    def mistakes(self, label: str = None, ground_truth: str = None):
        gt_label_column = self.config.label_column()

        if gt_label_column is None:
            raise ValueError(
                "Cannot compute mistakes without ground truth label column"
            )

        filtered_df = self.df[
            self.df[self.prefix + "label"] != self.df[gt_label_column]
        ]

        if label:
            filtered_df = filtered_df[filtered_df[self.prefix + "label"] == label]

        if ground_truth:
            filtered_df = filtered_df[filtered_df[gt_label_column] == ground_truth]

        return AutolabelDataset(filtered_df, self.config)

    def correct(self):
        gt_label_column = self.config.label_column()

        if gt_label_column is None:
            raise ValueError("Cannot compute correct without ground truth label column")

        filtered_df = self.df[
            self.df[self.prefix + "label"] == self.df[gt_label_column]
        ]
        return AutolabelDataset(filtered_df, self.config)

    def correct_and_confident(self, threshold: float = 0.5):
        gt_label_column = self.config.label_column()

        if gt_label_column is None:
            raise ValueError("Cannot compute correct without ground truth label column")

        if not self.config.confidence():
            raise ValueError(
                "Cannot compute correct and confident without confidence scores"
            )

        filtered_df = self.df[
            (self.df[self.prefix + "label"] == self.df[gt_label_column])
            & (self.df[self.prefix + "confidence"] >= threshold)
        ]
        return AutolabelDataset(filtered_df, self.config)

    def eval(self):
        gt_label_column = self.config.label_column()

        if gt_label_column is None:
            raise ValueError("Cannot compute eval without ground truth label column")

        gt_labels = self.df[gt_label_column]
        llm_labels = [
            LLMAnnotation(**json.loads(x))
            for x in self.df[self.prefix + "annotation"].tolist()
        ]

        task = TaskFactory.from_config(self.config)

        metrics = task.eval(llm_labels, gt_labels)

        table = {}
        for metric in metrics:
            if not isinstance(metric.value, list):
                table[metric.name] = metric.value

        print_table(table, console=Console(), default_style=METRIC_TABLE_STYLE)

        return metrics

    def columns(self):
        """Return columns"""
        return self.df.columns.tolist()

    def _validate(self):
        """Validate Data"""
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


class DataValidationFailed(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
