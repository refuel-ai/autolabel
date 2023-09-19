from typing import List
import logging

from sklearn.metrics import roc_auc_score
import numpy as np

from autolabel.metrics import BaseMetric
from autolabel.schema import LLMAnnotation, MetricResult, MetricType

logger = logging.getLogger(__name__)


class AUROCMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute(
        self, llm_labels: List[LLMAnnotation], gt_labels: List[str]
    ) -> List[MetricResult]:
        if not gt_labels:
            logger.warning(
                "No ground truth labels were provided. Skipping AUROC metric."
            )
            return []

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

        if np.unique(match).shape[0] == 1:
            # all labels are the same
            auroc = 1 if match[0] == 1 else 0
        else:
            auroc = roc_auc_score(match, confidence)

        value = [
            MetricResult(
                name=MetricType.AUROC,
                value=auroc,
            )
        ]
        return value
