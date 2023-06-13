from typing import Dict, List

import logging

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain.prompts.example_selector.base import BaseExampleSelector

from .fixed_example_selector import FixedExampleSelector
from .vector_store import VectorStoreWrapper
from autolabel.configs import AutolabelConfig
from autolabel.schema import FewShotAlgorithm

ALGORITHM_TO_IMPLEMENTATION: Dict[FewShotAlgorithm, BaseExampleSelector] = {
    FewShotAlgorithm.FIXED: FixedExampleSelector,
    FewShotAlgorithm.SEMANTIC_SIMILARITY: SemanticSimilarityExampleSelector,
    FewShotAlgorithm.MAX_MARGINAL_RELEVANCE: MaxMarginalRelevanceExampleSelector,
}

logger = logging.getLogger(__name__)


class ExampleSelectorFactory:
    CANDIDATE_EXAMPLES_FACTOR = 5
    MAX_CANDIDATE_EXAMPLES = 100

    @staticmethod
    def initialize_selector(
        config: AutolabelConfig, examples: List[Dict], columns: List[str]
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
        ]:
            params["embeddings"] = OpenAIEmbeddings()
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

        example_cls = ALGORITHM_TO_IMPLEMENTATION[algorithm]
        return example_cls.from_examples(**params)
