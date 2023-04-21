from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Metric(str, Enum):
    # Task agnostic
    SUPPORT = "support"
    COMPLETION_RATE = "completion_rate"
    # Classification metrics
    ACCURACY = "accuracy"
    CONFUSION_MATRIX = "confusion_matrix"
    LABEL_DISTRIBUTION = "label_distribution"
    F1 = "f1"
    # Confidence metrics
    AUROC = "auroc"
    THRESHOLD = "threshold"


class MetricResult(BaseModel):
    metric_type: Metric
    name: str
    value: Any


class LLMAnnotation(BaseModel):
    successfully_labeled: str
    label: Any
    curr_sample: Optional[str] = ""
    confidence_score: Optional[float] = None
    generation_info: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = ""
    prompt: Optional[str] = ""


class Dataset(BaseModel):
    id: str
    input_file: str
    start_index: int
    end_index: int

    class Config:
        orm_mode = True


# All available LLM providers
class LLMProvider(str, Enum):
    openai = "openai"
    openai_chat = "openai_chat"
    anthropic = "anthropic"
    cohere = "cohere"
    huggingface = "huggingface"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    ENTITY_RECOGNITION = "entity_recognition"


class Task(BaseModel):
    id: str
    task_type: TaskType
    provider: LLMProvider
    model_name: str
    config: str

    class Config:
        orm_mode = True


class TaskStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ACTIVE = "active"
    PAUSED = "paused"


class TaskResult(BaseModel):
    id: Optional[str] = None
    task_id: str
    dataset_id: str
    current_index: int
    output_file: str
    status: TaskStatus
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True


class Annotation(BaseModel):
    id: Optional[str] = None
    index: int
    llm_annotation: Optional[LLMAnnotation] = None

    class Config:
        orm_mode = True
