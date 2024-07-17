from collections import defaultdict
import json
from autolabel.cache import BaseCache
from autolabel.transforms import BaseTransform
from langchain_community.utilities import SerpAPIWrapper
from typing import Dict, Any, List
import logging
import pandas as pd

from autolabel.transforms.schema import (
    TransformError,
    TransformErrorType,
    TransformType,
)

logger = logging.getLogger(__name__)


class RefuelSerpAPIWrapper(SerpAPIWrapper):
    DEFAULT_ORGANIC_RESULTS_KEYS = ["position", "title", "link", "snippet"]

    def __init__(self, search_engine=None, params=None, serpapi_api_key=None):
        super().__init__(
            search_engine=search_engine, params=params, serpapi_api_key=serpapi_api_key
        )

    async def arun(self, query: str, **kwargs: Any) -> Dict:
        """Run query through SerpAPI and parse result async."""
        return self._process_response(await self.aresults(query))

    def _process_response(self, res: Dict) -> Dict:
        """
        Processes the response from Serp API and returns the search results.
        """
        cleaned_res = {}
        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")
        if "knowledge_graph" in res.keys():
            cleaned_res["knowledge_graph"] = json.dumps(res["knowledge_graph"])
        if "organic_results" in res.keys():
            organic_results = list(
                map(
                    lambda result: dict(
                        filter(
                            lambda item: item[0] in self.DEFAULT_ORGANIC_RESULTS_KEYS,
                            result.items(),
                        )
                    ),
                    res["organic_results"],
                )
            )
            cleaned_res["organic_results"] = json.dumps(organic_results)
        return cleaned_res


class SerpApi(BaseTransform):
    COLUMN_NAMES = ["knowledge_graph_results", "organic_search_results"]

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
        query_columns: List[str],
        query_template: str,
        serp_api_key: str,
        serp_args: dict = DEFAULT_ARGS,
    ) -> None:
        super().__init__(cache, output_columns)
        self.query_columns = query_columns
        self.query_template = query_template
        self.serp_api_key = serp_api_key
        self.serp_args = serp_args
        self.serp_api_wrapper = RefuelSerpAPIWrapper(
            search_engine=None, params=self.serp_args, serpapi_api_key=self.serp_api_key
        )

    def name(self) -> str:
        return TransformType.WEB_SEARCH_SERP_API

    async def _get_result(self, query):
        """
        Makes a request to Serp API with the query
        and returns the search results.
        """
        try:
            search_result = await self.serp_api_wrapper.arun(query=query)
        except Exception as e:
            logger.error(f"Error while making request to Serp API: {str(e)}")
            raise TransformError(
                TransformErrorType.TRANSFORM_API_ERROR,
                f"Error while making request with query: {query}",
            )
        return search_result

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        for col in self.query_columns:
            if col not in row:
                logger.error(
                    f"Missing query column: {col} in row {row}",
                )
        query = self.query_template.format_map(
            defaultdict(str, {key: val for key, val in row.items() if val is not None})
        )
        search_result = self.NULL_TRANSFORM_TOKEN
        if pd.isna(query) or query == self.NULL_TRANSFORM_TOKEN:
            raise TransformError(
                TransformErrorType.INVALID_INPUT,
                f"Empty query in row {row}",
            )
        else:
            search_result = await self._get_result(query)
        transformed_row = {
            self.output_columns["knowledge_graph_results"]: search_result.get(
                "knowledge_graph"
            ),
            self.output_columns["organic_search_results"]: search_result.get(
                "organic_results"
            ),
        }

        return self._return_output_row(transformed_row)

    def params(self):
        return {
            "query_columns": self.query_columns,
            "query_template": self.query_template,
            "output_columns": self.output_columns,
            "serp_api_key": self.serp_api_key,
            "serp_args": self.serp_args,
        }

    def input_columns(self) -> List[str]:
        return self.query_columns
