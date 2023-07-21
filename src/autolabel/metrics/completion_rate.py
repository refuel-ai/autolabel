from typing import List

from autolabel.metrics import BaseMetric
from autolabel.schema import LLMAnnotation, MetricResult, MetricType


class CompletionRateMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute(
        self, llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        completed = 0
        for label in llm_labels:
            if label.error is None:
                completed += 1

        if len(llm_labels) > 0:
            completion_rate = completed / len(llm_labels)
        else:
            completion_rate = 0.0
        value = [
            MetricResult(
                name=MetricType.COMPLETION_RATE,
                value=completion_rate,
            )
        ]
        return value
