from typing import List

from sklearn.metrics import accuracy_score

from autolabel.metrics import BaseMetric
from autolabel.schema import LLMAnnotation, MetricResult, MetricType


class AccuracyMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute(
        self, llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        filtered_llm_labels = []
        filtered_gt_labels = []
        for llm_label, gt_label in zip(llm_labels, gt_labels):
            if llm_label.error is None and gt_label != "nan":
                filtered_llm_labels.append(llm_label)
                filtered_gt_labels.append(gt_label)

        filtered_llm_labels = [llm_label.label for llm_label in filtered_llm_labels]

        if len(filtered_gt_labels) > 0:
            accuracy = accuracy_score(filtered_gt_labels, filtered_llm_labels)
        else:
            accuracy = 0.0

        value = [
            MetricResult(
                name=MetricType.ACCURACY,
                value=accuracy,
            )
        ]
        return value
