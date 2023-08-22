from autolabel.cache import BaseCache
from autolabel.transforms import BaseTransform
from langchain.utilities import SerpAPIWrapper
from typing import Dict, Any
import logging
import pandas as pd

from autolabel.transforms.schema import (
    TransformError,
    TransformErrorType,
    TransformType,
)

logger = logging.getLogger(__name__)


class SerpApi(BaseTransform):
    COLUMN_NAMES = [
        "result_column",
    ]

    DEFAULT_ARGS = {
        "engine": "google",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
    }

    def __init__(
        self,
        cache: BaseCache,
        output_columns: Dict[str, Any],
        query_column: str,
        serp_api_key: str,
        serp_args: dict = DEFAULT_ARGS,
    ) -> None:
        super().__init__(cache, output_columns)
        self.query_column = query_column
        self.serp_api_key = serp_api_key
        self.serp_args = serp_args
        self.serp_api_wrapper = SerpAPIWrapper(
            search_engine=None, params=self.serp_args, serpapi_api_key=self.serp_api_key
        )

    def name(self) -> str:
        return TransformType.SERP_API

    async def _get_result(self, query):
        """
        Makes a request to Serp API with the query
        and returns the search results.
        """
        try:
            search_result = await self.serp_api_wrapper.arun(query=query)
        except Exception as e:
            raise TransformError(
                TransformErrorType.SERP_API_ERROR,
                f"Error while making request to Serp API: {e}",
            )
        return search_result

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row[self.query_column]
        search_result = {}
        if pd.isna(query):
            logger.warning(f"Empty query in row {row}")
        else:
            search_result = await self._get_result(query)

        transformed_row = {self.output_columns["result_column"]: search_result}

        return self._return_output_row(transformed_row)

    def params(self):
        return {
            "query_column": self.query_column,
            "output_columns": self.output_columns,
            "serp_api_key": self.serp_api_key,
            "serp_args": self.serp_args,
        }
