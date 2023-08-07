from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTransform(ABC):
    def __init__(self, output_columns: Dict[str, Any]) -> None:
        super().__init__()
        self._output_columns = output_columns

    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def output_columns(self) -> Dict[str, Any]:
        return self._output_columns

    @abstractmethod
    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        pass

    async def apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return await self._apply(row)
