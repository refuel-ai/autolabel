import logging
from typing import Dict, List

from autolabel.configs import AutolabelConfig
from autolabel.schema import FewShotAlgorithm, ModelProvider
from langchain.embeddings import (
    CohereEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
    VertexAIEmbeddings,
)
from langchain.embeddings.base import Embeddings
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain.prompts.example_selector.base import BaseExampleSelector

from .fixed_example_selector import FixedExampleSelector
from .label_diversity_example_selector import (
    LabelDiversityRandomExampleSelector,
    LabelDiversitySimilarityExampleSelector,
)
from .vector_store import VectorStoreWrapper

ALGORITHM_TO_IMPLEMENTATION: Dict[FewShotAlgorithm, BaseExampleSelector] = {
    FewShotAlgorithm.FIXED: FixedExampleSelector,
    FewShotAlgorithm.SEMANTIC_SIMILARITY: SemanticSimilarityExampleSelector,
    FewShotAlgorithm.MAX_MARGINAL_RELEVANCE: MaxMarginalRelevanceExampleSelector,
    FewShotAlgorithm.LABEL_DIVERSITY_RANDOM: LabelDiversityRandomExampleSelector,
    FewShotAlgorithm.LABEL_DIVERSITY_SIMILARITY: LabelDiversitySimilarityExampleSelector,
}

DEFAULT_EMBEDDING_PROVIDER = OpenAIEmbeddings

PROVIDER_TO_MODEL: Dict[ModelProvider, Embeddings] = {
    ModelProvider.OPENAI: OpenAIEmbeddings,
    ModelProvider.GOOGLE: VertexAIEmbeddings,
    ModelProvider.HUGGINGFACE_PIPELINE: HuggingFaceEmbeddings,
    ModelProvider.COHERE: CohereEmbeddings,
}

logger = logging.getLogger(__name__)


class ExampleSelectorFactory:
    CANDIDATE_EXAMPLES_FACTOR = 5
    MAX_CANDIDATE_EXAMPLES = 100

    @staticmethod
    def initialize_selector(
        config: AutolabelConfig,
        examples: List[Dict],
        columns: List[str],
        cache: bool = True,
    ) -> BaseExampleSelector:
        algorithm = config.few_shot_algorithm()
        if not algorithm:
            return None
        try:
            algorithm = FewShotAlgorithm(algorithm)
        except ValueError as e:
            logger.error(
                f"{algorithm} is not in the list of supported few-shot algorithms: \
                {ALGORITHM_TO_IMPLEMENTATION.keys()}"
            )
            return None

        num_examples = config.few_shot_num_examples()
        params = {"examples": examples, "k": num_examples}
        if algorithm in [
            FewShotAlgorithm.SEMANTIC_SIMILARITY,
            FewShotAlgorithm.MAX_MARGINAL_RELEVANCE,
            FewShotAlgorithm.LABEL_DIVERSITY_SIMILARITY,
        ]:
            model_provider = config.embedding_provider()
            embedding_model_class = PROVIDER_TO_MODEL.get(
                model_provider, DEFAULT_EMBEDDING_PROVIDER
            )
            model_name = config.embedding_model_name()
            if model_name:
                embedding_model = embedding_model_class(model_name=model_name)
            else:
                embedding_model = embedding_model_class()
            params["embeddings"] = embedding_model
            params["vectorstore_cls"] = VectorStoreWrapper
            input_keys = [
                x
                for x in columns
                if x not in [config.label_column(), config.explanation_column()]
            ]
            params["input_keys"] = input_keys
        if algorithm == FewShotAlgorithm.MAX_MARGINAL_RELEVANCE:
            params["fetch_k"] = min(
                ExampleSelectorFactory.MAX_CANDIDATE_EXAMPLES,
                ExampleSelectorFactory.CANDIDATE_EXAMPLES_FACTOR * params["k"],
            )
        if algorithm in [
            FewShotAlgorithm.LABEL_DIVERSITY_RANDOM,
            FewShotAlgorithm.LABEL_DIVERSITY_SIMILARITY,
        ]:
            params["label_key"] = config.label_column()
            params["num_labels"] = len(config.labels_list())

        example_cls = ALGORITHM_TO_IMPLEMENTATION[algorithm]

        if algorithm not in [
            FewShotAlgorithm.FIXED,
            FewShotAlgorithm.LABEL_DIVERSITY_RANDOM,
        ]:
            params["cache"] = cache

        return example_cls.from_examples(**params)
