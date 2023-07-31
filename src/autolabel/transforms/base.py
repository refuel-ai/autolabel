from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseTransform(ABC):
    def __init__(self, output_columns: List[str]) -> None:
        super().__init__()
        self.output_columns = output_columns

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def transform(self, row: Dict[str, Any]) -> Dict[str, Any]:
        pass
