from typing import List

from autolabel.metrics import BaseMetric
from autolabel.schema import LLMAnnotation, MetricResult, MetricType


class SupportMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute(
        self, llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        value = [MetricResult(name=MetricType.SUPPORT, value=len(llm_labels))]
        return value
