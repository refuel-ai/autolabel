from typing import List
import logging
import pylcs


from sklearn.metrics import roc_auc_score
import numpy as np

from autolabel.metrics import BaseMetric
from autolabel.schema import LLMAnnotation, MetricResult, MetricType

logger = logging.getLogger(__name__)


class AUROCMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
    
    def similarity_acceptance(self, a, b) -> bool:
        if not isinstance(a, str) or not isinstance(b, str): 0
        a, b = a.replace('"', "'"), b.replace('"', "'")
        substring_lengths = pylcs.lcs_string_length(a, b)
        return substring_lengths / max(len(a) + 1e-5, len(b) + 1e-5)

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
            int(self.similarity_acceptance(llm_label.label, gt_label) > 0.95)
            for llm_label, gt_label in zip(filtered_llm_labels, filtered_gt_labels)
        ]
        confidence = [llm_label.confidence_score for llm_label in filtered_llm_labels]
        if np.unique(match).shape[0] == 1:
            # all labels are the same
            auroc = 1 if match[0] == 1 else 0
        elif len(match) > 0 and len(confidence) == len(match):
            auroc = roc_auc_score(match, confidence)
        else:
            auroc = 0

        value = [
            MetricResult(
                name=MetricType.AUROC,
                value=auroc,
            )
        ]
        return value
