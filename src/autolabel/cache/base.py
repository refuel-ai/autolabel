"""Base interface that all caches will implement."""

from abc import ABC, abstractmethod
from typing import Optional, List
from autolabel.schema import CacheEntry
from langchain.schema import Generation


class BaseCache(ABC):
    """used to store AutoLabeling results, allowing for interrupted labeling runs to be continued from a save point without the need to restart from the beginning. Any custom Cache classes should extend from BaseCache."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def lookup(self, entry: CacheEntry) -> List[Generation]:
        """abstract method to retrieve a cached entry. Must be implemented by classes derived from BaseCache."""
        pass

    @abstractmethod
    def update(self, entry: CacheEntry) -> None:
        """abstract method to update the cache with a new entry. Must be implemented by classes derived from BaseCache."""
        pass
