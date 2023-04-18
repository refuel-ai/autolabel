import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import (
    LengthBasedExampleSelector,
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector
from langchain.vectorstores import Chroma

from refuel_oracle.config import Config


# All available LLM providers
class ExampleSelectorStrategy(str, Enum):
    semantic_similarity = "semantic_similarity"
    n_gram_overlap = "n_gram_overlap"
    length_based = "length_based"
    maximal_marginal_relevance = "maximal_marginal_relevance"


class ExampleSelector:

    STRATEGY_TO_SELECTOR = {
        ExampleSelectorStrategy.semantic_similarity: SemanticSimilarityExampleSelector,
        ExampleSelectorStrategy.n_gram_overlap: NGramOverlapExampleSelector,
        ExampleSelectorStrategy.length_based: LengthBasedExampleSelector,
        ExampleSelectorStrategy.maximal_marginal_relevance: MaxMarginalRelevanceExampleSelector,
    }

    EXAMPLE_SELECTOR_STRATEGY_DEFAULT_PARAMS = {
        ExampleSelectorStrategy.semantic_similarity: {
            "vectorstore_cls": Chroma,
            "k": 3,
        },
        ExampleSelectorStrategy.n_gram_overlap: {"threshold": -1.0},
        ExampleSelectorStrategy.length_based: {"max_length": 25},
        ExampleSelectorStrategy.maximal_marginal_relevance: {
            "vectorstore_cls": Chroma,
            "k": 3,
        },
    }

    def __init__(self, config: Config) -> None:
        self.config = config.get("example_selector", {})
        self.examples = config.get("seed_examples", [])
        self.example_selector_strategy = self.config.get("name", None)
        example_selector_params = self.config.get("params", {})
        example_selector_class = ExampleSelector.STRATEGY_TO_SELECTOR[
            self.example_selector_strategy
        ]
        example_selector_params = (
            ExampleSelector.EXAMPLE_SELECTOR_STRATEGY_DEFAULT_PARAMS[
                self.example_selector_strategy
            ]
            | example_selector_params
        )
        example_selector_params["examples"] = self.examples
        example_selector_params["embeddings"] = OpenAIEmbeddings()
        self.example_selector = example_selector_class.from_examples(
            **example_selector_params
        )

    def is_example_selector(self) -> bool:
        if self.example_selector_strategy:
            return self.example_selector_strategy in [
                strategy.value for strategy in ExampleSelectorStrategy
            ]
        return False

    def get_examples(self, input):
        if self.is_example_selector():
            return self.example_selector.select_examples({"example": input})
        else:
            return self.examples
