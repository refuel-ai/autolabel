"""Base interface that all caches will implement."""

from abc import ABC, abstractmethod
from typing import Optional, List
from autolabel.schema import CacheEntry
from langchain.schema import Generation


class BaseCache(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def lookup(self, entry: CacheEntry) -> List[Generation]:
        pass

    @abstractmethod
    def update(self, entry: CacheEntry) -> None:
        pass
