from abc import ABC, abstractmethod
from typing import Dict, Any
from autolabel.cache import BaseCache
from autolabel.schema import TransformCacheEntry


class BaseTransform(ABC):
    """Base class for all transforms."""

    TTL_MS = 60 * 60 * 24 * 7 * 1000  # 1 week

    def __init__(self, output_columns: Dict[str, Any], cache: BaseCache) -> None:
        super().__init__()
        self._output_columns = output_columns
        self.cache = cache

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @property
    @abstractmethod
    def output_columns(self) -> Dict[str, Any]:
        return self._output_columns

    @abstractmethod
    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """
        Returns a dictionary of parameters that can be used to uniquely identify this transform.
        """
        return {}

    async def apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if self.cache is not None:
            cache_entry = TransformCacheEntry(
                transform_name=self.name(),
                transform_params=self.params(),
                input=row,
                ttl_ms=self.TTL_MS,
            )
            output = self.cache.lookup(cache_entry)

            if output is not None:
                # Cache hit
                return output

        output = await self._apply(row)

        if self.cache is not None:
            cache_entry.output = output
            self.cache.update(cache_entry)

        return output
