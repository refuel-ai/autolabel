from __future__ import annotations

import bisect
from collections.abc import Callable
from typing import Dict, List, Optional, Tuple, Union

from sqlalchemy.sql import text as sql_text

from autolabel.configs import AutolabelConfig
from autolabel.few_shot.vector_store import VectorStoreWrapper, cos_sim


class LabelSelector:
    """Returns the most similar labels to a given input. Used for
    classification tasks with a large number of possible classes."""

    labels: List[str]
    """A list of the possible labels to choose from."""

    label_descriptions: Optional[Dict[str, str]]
    """A dictionary of label descriptions. If provided, the selector will
    use these descriptions to find the most similar labels to the input."""

    labels_embeddings: Dict = {}
    """Dict used to store embeddings of each label"""

    cache: bool = True
    """Whether to cache the embeddings of labels"""

    def __init__(
        self,
        config: Union[AutolabelConfig, str, dict],
        embedding_func: Callable,
        cache: bool = True,
    ) -> None:
        self.config = config
        self.labels = self.config.labels_list()
        self.label_descriptions = self.config.label_descriptions()
        self.k = min(self.config.label_selection_count(), len(self.labels))
        self.threshold = self.config.label_selection_threshold()
        self.cache = cache
        print(self.k, self.threshold)
        self.vectorStore = VectorStoreWrapper(
            embedding_function=embedding_func, cache=self.cache
        )

        # Get the embeddings of the labels
        if self.label_descriptions is not None:
            (labels, descriptions) = zip(*self.label_descriptions.items())
            embeddings = self.vectorStore._get_embeddings(descriptions)
            for i, label in enumerate(labels):
                self.labels_embeddings[label] = embeddings[i]
        else:
            embeddings = self.vectorStore._get_embeddings(self.labels)
            for i, label in enumerate(labels):
                self.labels_embeddings[label] = embeddings[i]

    def select_labels(self, input: str) -> List[str]:
        """Select which labels to use based on the similarity to input"""
        input_embedding = self.vectorStore._get_embeddings([input])

        scores = []
        for label, embedding in self.labels_embeddings.items():
            similarity = cos_sim(embedding, input_embedding)
            # insert into scores, while maintaining sorted order
            bisect.insort(scores, (similarity, label))

        # remove labels with similarity score less than self.threshold*topScore
        return [
            label
            for (score, label) in scores[-self.k :]
            if score > self.threshold * scores[-1][0]
        ]
