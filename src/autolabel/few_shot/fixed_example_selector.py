from __future__ import annotations

from typing import Dict, List, Optional

from langchain.prompts.example_selector.base import BaseExampleSelector
from pydantic import BaseModel, Extra


class FixedExampleSelector(BaseExampleSelector, BaseModel):
    """Example selector to handle the case of fixed few-shot context
    i.e. every input prompt to the labeling model has the same few-shot examples
    """

    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    k: int = 4
    """Number of examples to select"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_example(self, example: Dict[str, str]) -> None:
        self.examples.append(example)

    def select_examples(
        self,
        input_variables: Dict[str, str],
        **kwargs,
    ) -> List[dict]:
        """Select which examples to use based on the input lengths."""
        selected_labels_map = kwargs.get("selected_labels_map")

        if not selected_labels_map:
            return self.examples[: self.k]

        # get the examples where label matches the selected labels
        valid_examples = []
        for example in self.examples:
            valid = True
            for label_column, selected_labels in selected_labels_map.items():
                if example.get(label_column) not in selected_labels:
                    valid = False
                    break
            if valid:
                valid_examples.append(example)

        return valid_examples[: min(self.k, len(valid_examples))]

    @classmethod
    def from_examples(
        cls,
        examples: List,
        k: int = 4,
    ) -> FixedExampleSelector:
        """Create pass-through example selector using example list

        Returns:
            The FixedExampleSelector instantiated
        """

        return cls(examples=examples, k=k)
