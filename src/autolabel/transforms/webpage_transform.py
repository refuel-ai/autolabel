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
from langchain_community.document_transformers import Html2TextTransformer
from langchain.docstore.document import Document
from autolabel.cache import BaseCache
from scrapingbee import ScrapingBeeClient

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BACKOFF = 2
DEFAULT_TIMEOUT = 60000  # in milliseconds


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
        }

    def name(self) -> str:
        return TransformType.WEBPAGE_TRANSFORM

    async def _load_url(self, url: str, retry_count=0) -> str:
        if retry_count >= self.max_retries:
            logger.warning(f"Max retries reached for URL: {url}")
            raise TransformError(
                TransformErrorType.MAX_RETRIES_REACHED, "Max retries reached"
            )

        try:
            response = self.client.get(url, params=self.scrapingbee_params)
            documents = [
                Document(page_content=response.content, metadata={"source": url})
            ]
            text = self.html2text_transformer.transform_documents(documents)[
                0
            ].page_content
            return text
        except Exception as e:
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
