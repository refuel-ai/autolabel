from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from langchain.schema import Generation
from pydantic import BaseModel

from autolabel.configs import AutolabelConfig
from autolabel.utils import calculate_md5


class ModelProvider(str, Enum):
    """Enum containing all LLM providers currently supported by autolabeler"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE_PIPELINE = "huggingface_pipeline"
    REFUEL = "refuel"
    GOOGLE = "google"
    COHERE = "cohere"


class TaskType(str, Enum):
    """Enum containing all the types of tasks that autolabeler currently supports"""

    CLASSIFICATION = "classification"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    QUESTION_ANSWERING = "question_answering"
    ENTITY_MATCHING = "entity_matching"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"


class FewShotAlgorithm(str, Enum):
    """Enum of supported algorithms for choosing which examples to provide the LLM in its instruction prompt"""

    FIXED = "fixed"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    MAX_MARGINAL_RELEVANCE = "max_marginal_relevance"
    LABEL_DIVERSITY_RANDOM = "label_diversity_random"
    LABEL_DIVERSITY_SIMILARITY = "label_diversity_similarity"


class TaskStatus(str, Enum):
    ACTIVE = "active"


class MetricType(str, Enum):
    """Enum of supported performance metrics. Some metrics are always available (task agnostic), while others are only supported by certain types of tasks"""

    # Task agnostic
    SUPPORT = "support"
    COMPLETION_RATE = "completion_rate"
    # Classification metrics
    ACCURACY = "accuracy"
    CONFUSION_MATRIX = "confusion_matrix"
    LABEL_DISTRIBUTION = "label_distribution"
    F1 = "f1"
    F1_MICRO = "f1_micro"
    F1_MACRO = "f1_macro"
    F1_WEIGHTED = "f1_weighted"
    TEXT_PARTIAL_MATCH = "text_partial_match"
    # Confidence metrics
    AUROC = "auroc"
    THRESHOLD = "threshold"


class F1Type(str, Enum):
    MULTI_LABEL = "multi_label"
    TEXT = "text"


class MetricResult(BaseModel):
    """Contains performance metrics gathered from autolabeler runs"""

    name: str
    value: Any


class ErrorType(str, Enum):
    """Enum of supported error types"""

    LLM_PROVIDER_ERROR = "llm_provider_error"
    PARSING_ERROR = "parsing_error"
    OUTPUT_GUIDELINES_NOT_FOLLOWED_ERROR = "output_guidelines_not_followed_error"
    EMPTY_RESPONSE_ERROR = "empty_response_error"


class LabelingError(BaseModel):
    """Contains information about an error that occurred during the labeling process"""

    error_type: ErrorType
    error_message: str


class LLMAnnotation(BaseModel):
    """Contains label information of a given data point, including the generated label, the prompt given to the LLM, and the LLMs response. Optionally includes a confidence_score if supported by the model"""

    successfully_labeled: bool
    label: Any
    curr_sample: Optional[str] = ""
    confidence_score: Optional[float] = None
    generation_info: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = ""
    explanation: Optional[str] = ""
    prompt: Optional[str] = ""
    error: Optional[LabelingError] = None


class Dataset(BaseModel):
    """Contains Dataset parameters, including input file path, indexes for state management (e.g. job batching and retries), and a unique ID"""

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
        """
        Generates a unique ID for the given Dataset configuration
        Args:
            dataset: either 1) input file name or 2) pandas Dataframe
            config:  AutolabelConfig object containing project settings
            start_index: index to begin labeling job at (used for job batching, retries, state management)
            max_items: number of data points to label, beginning at start_index

        Returns:
            filehash: a unique ID generated from an MD5 hash of the functions parameters
        """
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


class RefuelLLMResult(BaseModel):
    """List of generated outputs. This is a List[List[]] because
    each input could have multiple candidate generations."""

    generations: List[List[Generation]]

    """Errors encountered while running the labeling job"""
    errors: List[Optional[LabelingError]]

    """Costs incurred during the labeling job"""
    costs: Optional[List[float]] = []
