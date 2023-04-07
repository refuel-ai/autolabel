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
            "embeddings": OpenAIEmbeddings(),
            "k": 3,
        },
        ExampleSelectorStrategy.n_gram_overlap: {"threshold": -1.0},
        ExampleSelectorStrategy.length_based: {"max_length": 25},
        ExampleSelectorStrategy.maximal_marginal_relevance: {
            "vectorstore_cls": Chroma,
            "embeddings": OpenAIEmbeddings(),
            "k": 3,
        },
    }

    def __init__(self, config: Config) -> None:
        self.config = config.get("example_selector", {})
        self.examples = config.get("seed_examples", [])
        self.example_selector_strategy = self.config.get("name", None)

    def is_example_selector(self) -> bool:
        if self.example_selector_strategy:
            return self.example_selector_strategy in [
                member.value for member in ExampleSelectorStrategy
            ]
        return False

    def examples_list_to_dict(self, examples: List) -> Dict:
        return {example["example"]: example["output"] for example in examples}

    def examples_dict_to_list(self, examples_dict: Dict) -> List:
        return [
            {"example": key, "output": value} for key, value in examples_dict.items()
        ]

    def get_examples(self, input):
        if self.is_example_selector():
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
            example_selector = example_selector_class.from_examples(
                **example_selector_params
            )

            return example_selector.select_examples({"example": input})
