from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel

class Metric(str, Enum):
    SUPPORT = "support"
    COMPLETION_RATE = "completion_rate"
    ACCURACY = "accuracy"


class LLMAnnotation(BaseModel):
    prompt: str
    successfully_labeled: bool
    label: str
    confidence_score: Optional[float] = None
    generation_info: Optional[Dict[str, Any]] = None


