from __future__ import annotations

from typing import Dict, List
import bisect

from autolabel.few_shot.vector_store import cos_sim

from langchain.embeddings.openai import OpenAIEmbeddings


class LabelSelector:
    """Returns the most similar labels to a given input. Used for
    classification tasks with a large number of possible classes."""

    labels: List[str]
    """A list of the possible labels to choose from."""

    k: int = 10
    """Number of labels to select"""

    embedding_func = OpenAIEmbeddings()
    """Function used to generate embeddings of labels/input"""

    labels_embeddings: Dict = {}
    """Dict used to store embeddings of each label"""

    def __init__(
        self, labels: List[str], k: int = 10, embedding_func=OpenAIEmbeddings()
    ) -> None:
        self.labels = labels
        self.k = min(k, len(labels))
        self.embedding_func = embedding_func
        for l in self.labels:
            self.labels_embeddings[l] = self.embedding_func.embed_query(l)

    def select_labels(self, input: str) -> List[str]:
        """Select which labels to use based on the similarity to input"""
        input_embedding = self.embedding_func.embed_query(input)

        scores = []
        for label, embedding in self.labels_embeddings.items():
            similarity = cos_sim(embedding, input_embedding)
            # insert into scores, while maintaining sorted order
            bisect.insort(scores, (similarity, label))
        return [label for (_, label) in scores[-self.k :]]

    @classmethod
    def from_examples(
        cls,
        labels: List[str],
        k: int = 10,
        embedding_func=OpenAIEmbeddings(),
    ) -> LabelSelector:
        """Create pass-through label selector using given list of labels

        Returns:
            The LabelSelector instantiated
        """
        return cls(labels=labels, k=k, embedding_func=embedding_func)
