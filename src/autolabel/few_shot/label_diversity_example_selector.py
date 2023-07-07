"""Example selector that selects examples based on SemanticSimilarity."""
from __future__ import annotations

import math
from itertools import groupby
from operator import itemgetter
from typing import Any, Dict, List, Optional, Type

from langchain.embeddings.base import Embeddings
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Extra


def sorted_values(values: Dict[str, str]) -> List[Any]:
    """Return a list of values in dict sorted by key."""
    return [values[val] for val in sorted(values)]


class LabelDiversityRandomExampleSelector(BaseExampleSelector, BaseModel):
    """Example selector that selects examples based on label diversity at random."""

    examples: List[dict]
    """A list of the examples that the prompt template expects."""
    k: int = 4
    """Number of examples to select."""
    label_key: str
    """Name of the label column/key."""
    num_labels: int
    """Number of different labels."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_example(self, example: Dict[str, str]) -> None:
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        selected_examples = []
        sorted_examples = sorted(self.examples, key=itemgetter(self.label_key))
        num_examples_per_label = math.ceil(self.k / self.num_labels)
        for label, label_examples in groupby(
            sorted_examples, key=itemgetter(self.label_key)
        ):
            label_examples_list = list(label_examples)
            selected_examples.extend(label_examples_list[:num_examples_per_label])
        return selected_examples[: self.k]

    @classmethod
    def from_examples(
        cls,
        examples: List[dict],
        label_key: str,
        num_labels: int,
        k: int = 4,
    ) -> LabelDiversityRandomExampleSelector:
        """Create label diversity example selector using example list and embeddings.

        Args:
            examples: List of examples to use in the prompt.
            k: Number of examples to select per label
            label_key: Determines which variable corresponds to the example's label

        Returns:
            The ExampleSelector instantiated
        """
        return cls(k=k, examples=examples, label_key=label_key, num_labels=num_labels)


class LabelDiversitySimilarityExampleSelector(BaseExampleSelector, BaseModel):
    """ExampleSelector that selects examples based on label diversity, while choosing the most similar examples for each label"""

    vectorstore: VectorStore
    """VectorStore than contains information about examples."""
    k: int = 4
    """Number of examples to select."""
    input_keys: Optional[List[str]] = None
    """Optional keys to filter input to. If provided, the search is based on
    the input variables instead of all variables."""
    label_key: str
    """Name of the label column/key."""
    num_labels: int
    """Number of different labels."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_example(self, example: Dict[str, str]) -> str:
        """Add new example to vectorstore."""
        if self.input_keys:
            string_example = " ".join(
                sorted_values({key: example[key] for key in self.input_keys})
            )
        else:
            string_example = " ".join(sorted_values(example))
        ids = self.vectorstore.add_texts([string_example], metadatas=[example])
        return ids[0]

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on label diversity and semantic similarity."""
        # Get the docs with the highest similarity for each label.
        if self.input_keys:
            input_variables = {key: input_variables[key] for key in self.input_keys}
        query = " ".join(sorted_values(input_variables))
        num_examples_per_label = math.ceil(self.k / self.num_labels)
        example_docs = self.vectorstore.label_diversity_similarity_search(
            query, self.label_key, k=num_examples_per_label
        )
        # Get the examples from the metadata.
        # This assumes that examples are stored in metadata.
        examples = [dict(e.metadata) for e in example_docs]
        return examples[: self.k]

    @classmethod
    def from_examples(
        cls,
        examples: List[dict],
        label_key: str,
        embeddings: Embeddings,
        vectorstore_cls: Type[VectorStore],
        num_labels: int,
        k: int = 4,
        input_keys: Optional[List[str]] = None,
        **vectorstore_cls_kwargs: Any,
    ) -> LabelDiversitySimilarityExampleSelector:
        """Create k-shot example selector using example list and embeddings, taking both label diversity and semantic similarity into account.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            label_key: The column name corresponding to the label
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        if input_keys:
            string_examples = [
                " ".join(sorted_values({k: eg[k] for k in input_keys}))
                for eg in examples
            ]
        else:
            string_examples = [" ".join(sorted_values(eg)) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            input_keys=input_keys,
            label_key=label_key,
            num_labels=num_labels,
        )
