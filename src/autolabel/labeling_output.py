from typing import List, Optional
import pandas as pd

from autolabel.schema import MetricResult, LLMAnnotation
from autolabel.configs import AutolabelConfig


class LabelingOutput:
    config: AutolabelConfig
    df: pd.DataFrame
    metrics: List[MetricResult]

    def __init__(
        self,
        config: AutolabelConfig,
        df: pd.DataFrame,
        metrics: List[MetricResult],
        llm_labels: Optional[List[LLMAnnotation]] = None,
    ) -> None:
        self.config = config
        self.df = df
        self.metrics = metrics
        self.prefix = self.config.task_name() + "_"

        if llm_labels is not None:
            self.add_columns_to_df(llm_labels)

    def add_columns_to_df(self, llm_labels: List[LLMAnnotation]):
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
        if self.metrics is not None:
            for metric in self.metrics:
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

    def __repr__(self):
        if self.df is not None:
            return self.df.__repr__()

    def __str__(self):
        if self.df is not None:
            return self.df.__str__()

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

    def errors(self):
        filtered_df = self.df[self.df[self.prefix + "error"].notnull()]
        return LabelingOutput(
            self.config,
            filtered_df,
            self.metrics,
        )

    def non_errors(self):
        filtered_df = self.df[self.df[self.prefix + "error"].isnull()]
        return LabelingOutput(
            self.config,
            filtered_df,
            self.metrics,
        )

    def mistakes(self, label: str = None, ground_truth: str = None):
        gt_label_column = self.config().label_column()

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

        return LabelingOutput(
            self.config,
            filtered_df,
            self.metrics,
        )

    def correct(self):
        gt_label_column = self.config().label_column()

        if gt_label_column is None:
            raise ValueError("Cannot compute correct without ground truth label column")

        filtered_df = self.df[
            self.df[self.prefix + "label"] == self.df[gt_label_column]
        ]
        return LabelingOutput(
            self.config,
            filtered_df,
            self.metrics,
        )

    def correct_and_confident(self, threshold: float = 0.5):
        gt_label_column = self.config().label_column()

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
        return LabelingOutput(
            self.config,
            filtered_df,
            self.metrics,
        )
