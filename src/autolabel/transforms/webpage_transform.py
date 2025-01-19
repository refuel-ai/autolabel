import asyncio
import logging
from typing import Any, Dict
from urllib.parse import urlparse

import pandas as pd
import requests
from langchain.docstore.document import Document
from langchain_community.document_transformers import Html2TextTransformer
from scrapingbee import ScrapingBeeClient

from autolabel.cache import BaseCache
from autolabel.transforms import BaseTransform
from autolabel.transforms.schema import (
    TransformError,
    TransformErrorType,
    TransformType,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF = 2
DEFAULT_TIMEOUT = 60000  # in milliseconds
PREMIUM_PROXY_PARAM = "premium_proxy"
JS_SCENARIO = {
    "instructions": [
        {
            "infinite_scroll": {
                "max_count": 0,
                "delay": 1000,
            },
        },
    ],
}


class WebpageTransform(BaseTransform):
    COLUMN_NAMES = [
        "content_column",
    ]

    def __init__(
        self,
        cache: BaseCache,
        output_columns: Dict[str, Any],
        url_column: str,
        timeout: int = DEFAULT_TIMEOUT,
        scrapingbee_api_key: str = None,
        max_retries: int = MAX_RETRIES,
        v2_enabled: bool = False,
        v2_api_key: str = None,
    ) -> None:
        super().__init__(cache, output_columns)
        self.url_column = url_column
        self.max_retries = max_retries
        self.timeout = timeout
        self.html2text_transformer = Html2TextTransformer()
        self.api_key = scrapingbee_api_key
        self.client = ScrapingBeeClient(api_key=self.api_key)
        self.scrapingbee_params = {
            "timeout": self.timeout,
            "transparent_status_code": "True",
            "js_scenario": JS_SCENARIO,
        }
        self.v2_enabled = v2_enabled
        self.v2_api_key = v2_api_key

    def name(self) -> str:
        return TransformType.WEBPAGE_TRANSFORM

    def _load_url_v2(self, url: str) -> str:
        try:
            headers = {"Authorization": f"Bearer {self.v2_api_key}"}

            url = f"https://r.jina.ai/{url}"
            response = requests.get(url, headers=headers)
            return response.text
        except Exception as e:
            logger.warning(f"Error fetching content from URL: {url}. Exception: {e}")
            raise TransformError(
                TransformErrorType.TRANSFORM_ERROR,
                f"Error fetching content from URL: {url}. Exception: {e}",
            )

    # On error, retry fetching the URL with a premium proxy. Only use exponential backoff for certain status codes.
    async def _load_url(self, url: str, retry_count=0) -> str:
        if self.v2_enabled:
            return self._load_url_v2(url)
        if retry_count >= self.max_retries:
            logger.warning(f"Max retries reached for URL: {url}")
            raise TransformError(
                TransformErrorType.MAX_RETRIES_REACHED,
                f"Max retries reached for URL: {url}",
            )
        if retry_count > 0:
            logger.warning(f"Retrying scraping URL: {url} with premium proxy")
            self.scrapingbee_params[PREMIUM_PROXY_PARAM] = "True"

        try:
            response = self.client.get(url, params=self.scrapingbee_params)
            response.raise_for_status()
            documents = [
                Document(page_content=response.content, metadata={"source": url}),
            ]
            text = self.html2text_transformer.transform_documents(documents)[
                0
            ].page_content
            return text
        except Exception as e:
            logger.warning(f"Error fetching content from URL: {url}. Exception: {e}")
            if response.status_code in [408, 429, 504]:
                await asyncio.sleep(BACKOFF**retry_count)
            return await self._load_url(url, retry_count=retry_count + 1)

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        url = row[self.url_column]
        if pd.isna(url) or url == self.NULL_TRANSFORM_TOKEN:
            raise TransformError(
                TransformErrorType.INVALID_INPUT,
                f"Empty url in row {row}",
            )
        if not urlparse(url).scheme:
            url = f"https://{url}"
        url_response_text = await self._load_url(url)

        transformed_row = {
            self.output_columns["content_column"]: url_response_text,
        }
        return self._return_output_row(transformed_row)

    def params(self):
        return {
            "url_column": self.url_column,
            "output_columns": self.output_columns,
            "timeout": self.timeout,
            "scrapingbee_api_key": self.api_key,
        }

    def input_columns(self):
        return [self.url_column]
