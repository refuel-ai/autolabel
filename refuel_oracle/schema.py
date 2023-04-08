from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


class Metric(str, Enum):
    # Task agnostic
    SUPPORT = "support"
    COMPLETION_RATE = "completion_rate"
    # Classification metrics
    ACCURACY = "accuracy"
    CONFUSION_MATRIX = "confusion_matrix"
    LABEL_DISTRIBUTION = "label_distribution"
    RECALL = "recall"


class MetricResult(BaseModel):
    metric_type: Metric
    name: str
    value: Any


class LLMAnnotation(BaseModel):
    successfully_labeled: str
    label: str
    confidence_score: Optional[float] = None
    generation_info: Optional[Dict[str, Any]] = None
