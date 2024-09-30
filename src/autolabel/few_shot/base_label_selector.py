from abc import ABC, abstractmethod
from typing import List


class BaseLabelSelector(ABC):
    @abstractmethod
    def select_labels(self, input: str) -> List[str]:
        pass
