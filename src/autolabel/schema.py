from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List

from pydantic import BaseModel

from autolabel.configs import ModelConfig, DatasetConfig, TaskConfig
from autolabel.utils import calculate_md5
from langchain.schema import Generation


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE_PIPELINE = "huggingface_pipeline"
    REFUEL = "refuel"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    MULTI_CHOICE_QUESTION_ANSWERING = "multi_choice_question_answering"
    ENTITY_MATCHING = "entity_matching"


class TaskStatus(str, Enum):
    ACTIVE = "active"


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

    @classmethod
    def create_id(
        self,
        input_file: str,
        dataset_config: DatasetConfig,
        start_index: int,
        max_items: int,
    ):
        filehash = calculate_md5(
            [open(input_file, "rb"), dataset_config.config, start_index, max_items]
        )
        return filehash


class Task(BaseModel):
    id: str
    task_type: TaskType
    provider: LLMProvider
    model_name: str
    config: str

    class Config:
        orm_mode = True

    @classmethod
    def create_id(self, task_config: TaskConfig, llm_config: ModelConfig):
        filehash = calculate_md5([task_config.config, llm_config.config])
        return filehash


class TaskRun(BaseModel):
    id: Optional[str] = None
    created_at: datetime
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


class CacheEntry(BaseModel):
    model_name: str
    prompt: str
    model_params: str
    generations: Optional[List[Generation]] = None

    class Config:
        orm_mode = True
