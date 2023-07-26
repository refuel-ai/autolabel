from typing import List

from sklearn.metrics import roc_auc_score

from autolabel.metrics import BaseMetric
from autolabel.schema import LLMAnnotation, MetricResult, MetricType


class AUROCMetric(BaseMetric):
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

        match = [
            int(llm_label.label == gt_label)
            for llm_label, gt_label in zip(filtered_llm_labels, filtered_gt_labels)
        ]
        confidence = [llm_label.confidence_score for llm_label in filtered_llm_labels]

        auroc = roc_auc_score(match, confidence)

        value = [
            MetricResult(
                name=MetricType.AUROC,
                value=auroc,
            )
        ]
        return value
