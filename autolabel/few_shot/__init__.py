from typing import Dict, List

from autolabel.configs import TaskConfig
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain.prompts.example_selector.base import BaseExampleSelector
from loguru import logger

from .fixed_example_selector import FixedExampleSelector
from .vector_store import VectorStoreWrapper

STRATEGY_TO_IMPLEMENTATION: Dict[str, BaseExampleSelector] = {
    "fixed_few_shot": FixedExampleSelector,
    "semantic_similarity": SemanticSimilarityExampleSelector,
    "max_marginal_relevance": MaxMarginalRelevanceExampleSelector,
}


class ExampleSelectorFactory:
    DEFAULT_STRATEGY = "fixed_few_shot"
    DEFAULT_NUM_EXAMPLES = 4

    @staticmethod
    def initialize_selector(
        config: TaskConfig, examples: List[Dict]
    ) -> BaseExampleSelector:
        example_selector_config = config.get_example_selector()
        strategy = example_selector_config.get(
            "strategy", ExampleSelectorFactory.DEFAULT_STRATEGY
        )
        num_examples = example_selector_config.get(
            "num_examples", ExampleSelectorFactory.DEFAULT_NUM_EXAMPLES
        )

        if strategy not in STRATEGY_TO_IMPLEMENTATION:
            logger.error(
                f"Example selection: {strategy} is not in the list of supported strategies: {STRATEGY_TO_IMPLEMENTATION.keys()}"
            )
            return None

        params = {"examples": examples, "k": num_examples}
        if strategy in ["semantic_similarity", "max_marginal_relevance"]:
            params["embeddings"] = OpenAIEmbeddings()
            params["vectorstore_cls"] = VectorStoreWrapper

        example_cls = STRATEGY_TO_IMPLEMENTATION[strategy]
        return example_cls.from_examples(**params)
