from abc import ABC, abstractmethod
from autolabel.cache import BaseCache
from autolabel.transforms.schema import TransformCacheEntry, TransformError
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseTransform(ABC):
    """Base class for all transforms."""

    TTL_MS = 60 * 60 * 24 * 7 * 1000  # 1 week
    NULL_TRANSFORM_TOKEN = "NO_TRANSFORM"

    def __init__(self, cache: BaseCache, output_columns: Dict[str, Any]) -> None:
        """
        Initialize a transform.
        Args:
            cache: A cache object to use for caching the results of this transform.
            output_columns: A dictionary of output columns. The keys are the names of the output columns as expected by the transform. The values are the column names they should be mapped to in the dataset.
        """
        super().__init__()
        self._output_columns = output_columns
        self.cache = cache

    @staticmethod
    @abstractmethod
    def name() -> str:
        """
        Returns the name of the transform.
        """
        pass

    @property
    def output_columns(self) -> Dict[str, Any]:
        """
        Returns a dictionary of output columns. The keys are the names of the output columns
        as expected by the transform. The values are the column names they should be mapped to in
        the dataset.
        """
        return {k: self._output_columns.get(k, None) for k in self.COLUMN_NAMES}

    @property
    def transform_error_column(self) -> str:
        """
        Returns the name of the column that stores the error if transformation fails.
        """
        return f"{self.name()}_error"

    @abstractmethod
    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies the transform to the given row.
        Args:
            row: A dictionary representing a row in the dataset. The keys are the column names and the values are the column values.
        Returns:
            A dictionary representing the transformed row. The keys are the column names and the values are the column values.
        """
        pass

    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """
        Returns a dictionary of parameters that can be used to uniquely identify this transform.
        Returns:
            A dictionary of parameters that can be used to uniquely identify this transform.
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

        try:
            output = await self._apply(row)
        except Exception as e:
            logger.error(f"Error applying transform {self.name()}. Exception: {str(e)}")
            output = {
                k: self.NULL_TRANSFORM_TOKEN
                for k in self.output_columns.values()
                if k is not None
            }
            output[self.transform_error_column] = str(e)
            return output

        if self.cache is not None:
            cache_entry.output = output
            self.cache.update(cache_entry)

        return output

    def _return_output_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns the output row with the correct column names.
        Args:
            row: The output row.
        Returns:
            The output row with the correct column names.
        """
        # remove null key
        row.pop(None, None)
        return row
