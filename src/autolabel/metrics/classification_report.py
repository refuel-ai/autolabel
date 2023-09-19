from typing import List
import logging

from sklearn.metrics import classification_report

from autolabel.metrics import BaseMetric
from autolabel.schema import LLMAnnotation, MetricResult, MetricType

logger = logging.getLogger(__name__)


class ClassificationReportMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute(
        self, llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        # If there are not ground truth labels, return an empty list
        if not gt_labels:
            logger.warning(
                "No ground truth labels were provided. Skipping classification report metric."
            )
            return []

        filtered_llm_labels = []
        filtered_gt_labels = []
        for llm_label, gt_label in zip(llm_labels, gt_labels):
            if llm_label.error is None and gt_label != "nan":
                filtered_llm_labels.append(llm_label)
                filtered_gt_labels.append(gt_label)

        filtered_llm_labels = [llm_label.label for llm_label in filtered_llm_labels]

        # if there are no labels, return empty list
        if len(filtered_gt_labels) == 0:
            return []

        # if the number of unique labels is too large, return empty list
        if len(set(filtered_gt_labels + filtered_llm_labels)) > 10:
            return []

        report = classification_report(filtered_gt_labels, filtered_llm_labels)

        value = [
            MetricResult(
                name=MetricType.CLASSIFICATION_REPORT,
                value=report,
                show_running=False,
            )
        ]
        return value
