from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import json
import pandas as pd
from langchain.schema import Generation, ChatGeneration
from pydantic import BaseModel

from autolabel.configs import AutolabelConfig
from autolabel.utils import calculate_md5


class ModelProvider(str, Enum):
    """Enum containing all LLM providers currently supported by autolabeler"""

    OPENAI = "openai"
    OPENAI_VISION = "openai_vision"
    ANTHROPIC = "anthropic"
    HUGGINGFACE_PIPELINE = "huggingface_pipeline"
    HUGGINGFACE_PIPELINE_VISION = "huggingface_pipeline_vision"
    REFUEL = "refuel"
    GOOGLE = "google"
    COHERE = "cohere"
    CUSTOM = "custom"
    TGI = "tgi"


class TaskType(str, Enum):
    """Enum containing all the types of tasks that autolabeler currently supports"""

    CLASSIFICATION = "classification"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    QUESTION_ANSWERING = "question_answering"
    ENTITY_MATCHING = "entity_matching"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    ATTRIBUTE_EXTRACTION = "attribute_extraction"


class FewShotAlgorithm(str, Enum):
    """Enum of supported algorithms for choosing which examples to provide the LLM in its instruction prompt"""

    FIXED = "fixed"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    MAX_MARGINAL_RELEVANCE = "max_marginal_relevance"
    LABEL_DIVERSITY_RANDOM = "label_diversity_random"
    LABEL_DIVERSITY_SIMILARITY = "label_diversity_similarity"


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
    # Token Classification metrics
    F1_EXACT = "f1_exact"
    F1_STRICT = "f1_strict"
    F1_PARTIAL = "f1_partial"
    F1_ENT_TYPE = "f1_ent_type"
    # Confidence metrics
    AUROC = "auroc"
    THRESHOLD = "threshold"

    # Aggregate Metrics
    CLASSIFICATION_REPORT = "classification_report"


class F1Type(str, Enum):
    MULTI_LABEL = "multi_label"
    TEXT = "text"


class MetricResult(BaseModel):
    """Contains performance metrics gathered from autolabeler runs"""

    name: str
    value: Any
    show_running: Optional[bool] = True


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
    curr_sample: Optional[bytes] = ""
    confidence_score: Optional[float] = None
    generation_info: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = ""
    explanation: Optional[str] = ""
    prompt: Optional[str] = ""
    confidence_prompt: Optional[str] = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost: Optional[float] = None
    latency: Optional[float] = None
    error: Optional[LabelingError] = None


class GenerationCacheEntry(BaseModel):
    model_name: str
    prompt: str
    model_params: str
    generations: Optional[List[Union[Generation, ChatGeneration]]] = None
    creation_time_ms: Optional[int] = -1
    ttl_ms: Optional[int] = -1

    class Config:
        orm_mode = True

    def get_id(self) -> str:
        """
        Generates a unique ID for the given generation cache configuration
        """
        return calculate_md5([self.model_name, self.model_params, self.prompt])

    def get_serialized_output(self) -> str:
        """
        Returns the serialized cache entry output
        """
        return json.dumps([gen.dict() for gen in self.generations])

    def deserialize_output(
        self, output: str
    ) -> List[Union[Generation, ChatGeneration]]:
        """
        Deserializes the cache entry output
        """
        generations = [
            Generation(**gen) if gen["type"] == "Generation" else ChatGeneration(**gen)
            for gen in json.loads(output)
        ]
        return generations


class ConfidenceCacheEntry(BaseModel):
    prompt: Optional[str] = ""
    raw_response: Optional[str] = ""
    logprobs: Optional[list] = None
    score_type: Optional[str] = "logprob_average"
    creation_time_ms: Optional[int] = -1
    ttl_ms: Optional[int] = -1

    class Config:
        orm_mode = True

    def get_id(self) -> str:
        """
        Generates a unique ID for the given confidence cache configuration
        """
        return calculate_md5([self.prompt, self.raw_response, self.score_type])

    def get_serialized_output(self) -> str:
        """
        Returns the serialized cache entry output
        """
        return json.dumps(self.logprobs)

    def deserialize_output(self, output: str) -> Dict[str, float]:
        """
        Deserializes the cache entry output
        """
        return json.loads(output)


class RefuelLLMResult(BaseModel):
    """List of generated outputs. This is a List[List[]] because
    each input could have multiple candidate generations."""

    generations: List[List[Union[Generation, ChatGeneration]]]

    """Errors encountered while running the labeling job"""
    errors: List[Optional[LabelingError]]

    """Costs incurred during the labeling job"""
    costs: Optional[List[float]] = []

    """Latencies incurred during the labeling job"""
    latencies: Optional[List[float]] = []


class AggregationFunction(str, Enum):
    """Enum of supported aggregation functions"""

    MAX = "max"
    MEAN = "mean"


AUTO_CONFIDENCE_CHUNKING_COLUMN = "auto"
TASK_CHAIN_TYPE = "task_chain"
