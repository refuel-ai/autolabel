from __future__ import annotations

import bisect
from collections.abc import Callable
from typing import Dict, List, Optional, Tuple, Union

import torch
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
        self.k = min(self.config.max_selected_labels(), len(self.labels))
        self.threshold = self.config.label_selection_threshold()
        self.cache = cache
        self.vectorStore = VectorStoreWrapper(
            embedding_function=embedding_func, cache=self.cache
        )

        # Get the embeddings of the labels
        if self.label_descriptions is not None:
            (labels, descriptions) = zip(*self.label_descriptions.items())
            self.labels = list(labels)
            self.labels_embeddings = torch.Tensor(
                self.vectorStore._get_embeddings(descriptions)
            )
        else:
            self.labels_embeddings = torch.Tensor(
                self.vectorStore._get_embeddings(self.labels)
            )
        print(type(self.labels_embeddings))

    def select_labels(self, input: str) -> List[str]:
        """Select which labels to use based on the similarity to input"""
        input_embedding = torch.Tensor(self.vectorStore._get_embeddings([input]))
        scores = cos_sim(input_embedding, self.labels_embeddings).view(-1)
        scores = list(zip(scores, self.labels))
        scores.sort(key=lambda x: x[0])

        # remove labels with similarity score less than self.threshold*topScore
        return [
            label
            for (score, label) in scores[-self.k :]
            if score > self.threshold * scores[-1][0]
        ]
