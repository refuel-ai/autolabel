from __future__ import annotations

from pydantic import BaseModel, Extra

from typing import Dict, List
from langchain.prompts.example_selector.base import BaseExampleSelector


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

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the input lengths."""
        return self.examples[: self.k]

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
