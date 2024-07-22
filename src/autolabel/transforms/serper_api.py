from collections import defaultdict
import json
from autolabel.cache import BaseCache
from autolabel.transforms import BaseTransform
from langchain_community.utilities import GoogleSerperAPIWrapper
from typing import Dict, Any, List
import logging
import pandas as pd

from autolabel.transforms.schema import (
    TransformError,
    TransformErrorType,
    TransformType,
)

logger = logging.getLogger(__name__)


class RefuelSerperAPIWrapper(GoogleSerperAPIWrapper):
    DEFAULT_ORGANIC_RESULTS_KEYS = ["position", "title", "link", "snippet"]

    def __init__(self, params=None, serper_api_key=None):
        super().__init__(
            params=params,
            serper_api_key=serper_api_key,
        )

    async def arun(self, query: str, **kwargs: Any) -> Dict:
        """Run query through Serper.dev API and parse result async."""
        return self._process_response(await self.aresults(query))

    def _process_response(self, res: Dict) -> Dict:
        """
        Processes the response from Serper.dev API and returns the search results.
        """
        cleaned_res = {}
        if "error" in res.keys():
            raise ValueError(f"Got error from Serper.dev API: {res['error']}")
        if "knowledgeGraph" in res.keys():
            cleaned_res["knowledge_graph"] = json.dumps(res["knowledgeGraph"])
        if "organic" in res.keys():
            organic_results = list(
                map(
                    lambda result: dict(
                        filter(
                            lambda item: item[0] in self.DEFAULT_ORGANIC_RESULTS_KEYS,
                            result.items(),
                        )
                    ),
                    res["organic"],
                )
            )
            cleaned_res["organic_results"] = json.dumps(organic_results)
        return cleaned_res


class SerperApi(BaseTransform):
    COLUMN_NAMES = ["knowledge_graph_results", "organic_search_results"]

    DEFAULT_ARGS = {"gl": "us", "hl": "en"}

    def __init__(
        self,
        cache: BaseCache,
        output_columns: Dict[str, Any],
        query_columns: List[str],
        query_template: str,
        serper_api_key: str,
        serper_args: dict = DEFAULT_ARGS,
    ) -> None:
        super().__init__(cache, output_columns)
        self.query_columns = query_columns
        self.query_template = query_template
        self.serper_api_key = serper_api_key
        self.serper_args = serper_args
        self.serper_api_wrapper = RefuelSerperAPIWrapper(
            params=self.serper_args,
            serper_api_key=self.serper_api_key,
        )

    def name(self) -> str:
        return TransformType.WEB_SEARCH_SERPER

    async def _get_result(self, query):
        """
        Makes a request to Serper.dev API with the query
        and returns the search results.
        """
        try:
            search_result = await self.serper_api_wrapper.arun(query=query)
        except Exception as e:
            logger.error(f"Error while making request to Serper API: {str(e)}")
            raise TransformError(
                TransformErrorType.TRANSFORM_API_ERROR,
                f"Error while making request with query: {query}",
            )
        return search_result

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        for col in self.query_columns:
            if col not in row:
                logger.warning(
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
            "serper_api_key": self.serper_api_key,
            "serper_args": self.serper_args,
        }

    def input_columns(self) -> List[str]:
        return self.query_columns
