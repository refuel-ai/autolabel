"""Base interface that all caches will implement."""

from abc import ABC, abstractmethod


class BaseCache(ABC):

    """used to store AutoLabeling results, allowing for interrupted labeling runs to be continued from a save point without the need to restart from the beginning. Any custom Cache classes should extend from BaseCache."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def initialize():
        """Initialize the cache. Must be implemented by classes derived from BaseCache."""

    @abstractmethod
    def lookup(self, entry):
        """Abstract method to retrieve a cached entry. Must be implemented by classes derived from BaseCache."""

    @abstractmethod
    def update(self, entry):
        """Abstract method to update the cache with a new entry. Must be implemented by classes derived from BaseCache."""

    @abstractmethod
    def clear(self) -> None:
        """Abstract method to clear the cache. Must be implemented by classes derived from BaseCache."""
