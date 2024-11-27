from __future__ import annotations

from collections.abc import Callable
from typing import Dict, List, Optional, Union

from autolabel.configs import AutolabelConfig
from autolabel.few_shot.base_label_selector import BaseLabelSelector
from autolabel.few_shot.vector_store import VectorStoreWrapper


class LabelSelector(BaseLabelSelector):

    """
    Returns the most similar labels to a given input. Used for
    classification tasks with a large number of possible classes.
    """

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
        self.label_selection_attribute = self.config.label_selection_attribute()
        self.labels = []
        self.label_descriptions = None

        attributes = self.config.attributes()
        matching_attribute = next(
            (
                attr
                for attr in attributes
                if attr["name"] == self.label_selection_attribute
            ),
            None,
        )

        if matching_attribute is None:
            raise ValueError(
                f"No attribute found with name '{self.label_selection_attribute}'",
            )

        self.labels = matching_attribute.get("options", [])
        if not self.labels:
            raise ValueError(
                f"Attribute '{self.label_selection_attribute}' does not have any options",
            )

        self.k = min(self.config.max_selected_labels(), len(self.labels))
        self.cache = cache
        self.vectorStore = VectorStoreWrapper(
            embedding_function=embedding_func, cache=self.cache,
        )

        self.vectorStore.add_texts(self.labels)

    def select_labels(self, input: str) -> List[str]:
        """Select which labels to use based on the similarity to input"""
        documents = self.vectorStore.similarity_search(input, k=self.k)
        return [doc.page_content for doc in documents]
