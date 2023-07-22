from abc import ABC, abstractmethod
from typing import List

from autolabel.schema import LLMAnnotation, MetricResult


class BaseMetric(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def compute(
        self, llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        pass
