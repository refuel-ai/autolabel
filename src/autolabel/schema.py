from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List, Union
import pandas as pd

from pydantic import BaseModel

from autolabel.configs import AutolabelConfig
from autolabel.utils import calculate_md5
from langchain.schema import Generation


class ModelProvider(str, Enum):
    """An Enum Class containing all LLM providers currently supported by autolabeler"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE_PIPELINE = "huggingface_pipeline"
    REFUEL = "refuel"
    GOOGLE = "google"


class TaskType(str, Enum):
    """An Enum Class containing all the types of tasks that autolabeler currently supports"""

    CLASSIFICATION = "classification"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    MULTI_CHOICE_QUESTION_ANSWERING = "multi_choice_question_answering"
    ENTITY_MATCHING = "entity_matching"


class FewShotAlgorithm(str, Enum):
    """An Enum Class containing the algorithms currently supported for choosing which examples to provide the LLM in its instruction prompt"""

    FIXED = "fixed"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    MAX_MARGINAL_RELEVANCE = "max_marginal_relevance"


class TaskStatus(str, Enum):
    ACTIVE = "active"


class Metric(str, Enum):
    """An Enum Class containing all possible ways of measuring autolabeler performance. Some metrics are always available (task agnostic), while others are only supported by certain types of Tasks"""

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
    """An Object for storing performance metrics gathered from autolabeler runs"""

    metric_type: Metric
    name: str
    value: Any


class LLMAnnotation(BaseModel):
    """An Object for storing the generated label information and metadata for a given data point. Contains useful debugging information including the prompt used to generate this label and the model's confidence in its answer (for models that support confidence)"""

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
        dataset: Union[str, pd.DataFrame],
        config: AutolabelConfig,
        start_index: int,
        max_items: int,
    ) -> str:
        if isinstance(dataset, str):
            filehash = calculate_md5(
                [open(dataset, "rb"), config._dataset_config, start_index, max_items]
            )
        else:
            filehash = calculate_md5(
                [dataset.to_csv(), config._dataset_config, start_index, max_items]
            )
        return filehash


class Task(BaseModel):
    id: str
    task_type: TaskType
    model_name: str
    config: str

    class Config:
        orm_mode = True

    @classmethod
    def create_id(self, config: AutolabelConfig) -> str:
        filehash = calculate_md5(config.config)
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
