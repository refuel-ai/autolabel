from urllib.parse import urlparse
from autolabel.transforms.schema import (
    TransformType,
    TransformError,
    TransformErrorType,
)
from autolabel.transforms import BaseTransform
from typing import Dict, Any
import asyncio
import logging
import pandas as pd
import ssl
import json
from langchain_community.document_transformers import Html2TextTransformer
from langchain.docstore.document import Document
from autolabel.cache import BaseCache
from scrapingbee import ScrapingBeeClient

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BACKOFF = 2
DEFAULT_TIMEOUT = 60000  # in milliseconds
PREMIUM_PROXY_PARAM = "premium_proxy"
JS_SCENARIO = {
    "instructions": [
        {
            "infinite_scroll": {
                "max_count": 0,
                "delay": 1000,
            }
        }
    ]
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
    ) -> None:
        super().__init__(cache, output_columns)
        self.url_column = url_column
        self.max_retries = MAX_RETRIES
        self.timeout = DEFAULT_TIMEOUT
        self.html2text_transformer = Html2TextTransformer()
        self.api_key = scrapingbee_api_key
        self.client = ScrapingBeeClient(api_key=self.api_key)
        self.scrapingbee_params = {
            "timeout": self.timeout,
            "transparent_status_code": "True",
        }

    def name(self) -> str:
        return TransformType.WEBPAGE_TRANSFORM

    # On error, retry fetching the URL with a premium proxy. Only use exponential backoff for certain status codes.
    async def _load_url(self, url: str, retry_count=0) -> str:
        if (
            retry_count >= self.max_retries
            or self.scrapingbee_params.get(PREMIUM_PROXY_PARAM) == "True"
        ):
            logger.warning(f"Max retries reached for URL: {url}")
            raise TransformError(
                TransformErrorType.MAX_RETRIES_REACHED, "Max retries reached"
            )
        if retry_count > 0:
            logger.warning(f"Retrying scraping URL: {url} with premium proxy")
            self.scrapingbee_params[PREMIUM_PROXY_PARAM] = "True"
            self.scrapingbee_params["js_scenario"] = JS_SCENARIO

        try:
            response = self.client.get(url, params=self.scrapingbee_params)
            response.raise_for_status()
            documents = [
                Document(page_content=response.content, metadata={"source": url})
            ]
            text = self.html2text_transformer.transform_documents(documents)[
                0
            ].page_content
            return text
        except Exception as e:
            logger.warning(f"Error fetching content from URL: {url}. Exception: {e}")
            if response.status_code in [408, 425, 429, 500, 502, 503, 504]:
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
