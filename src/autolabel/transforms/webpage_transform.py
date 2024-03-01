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
from langchain_community.document_loaders import SeleniumURLLoader
from autolabel.cache import BaseCache

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
    ) -> None:
        super().__init__(cache, output_columns)
        self.url_column = url_column
        self.max_retries = MAX_RETRIES
        self.timeout = timeout
        self.html2text_transformer = Html2TextTransformer()

    def name(self) -> str:
        return TransformType.WEBPAGE_TRANSFORM

    async def _load_url(self, url: str, retry_count=0) -> str:
        if retry_count >= self.max_retries:
            logger.warning(f"Max retries reached for URL: {url}")
            raise TransformError(
                TransformErrorType.MAX_RETRIES_REACHED, "Max retries reached"
            )

        try:
            loader = SeleniumURLLoader(urls=[url])
            documents = loader.load()
            text = self.html2text_transformer.transform_documents(documents)[
                0
            ].page_content
            return text
        except ssl.SSLCertVerificationError as e:
            logger.warning(
                f"SSL verification error when fetching content from URL: {url}, retrying with verify=False"
            )
            await asyncio.sleep(BACKOFF**retry_count)
            return await self._load_url(url, retry_count=retry_count + 1)
        except Exception as e:
            logger.error(f"Error fetching content from URL: {url}. Exception: {e}")
            raise e

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        url = row[self.url_column]
        if pd.isna(url) or url == self.NULL_TRANSFORM_TOKEN:
            raise TransformError(
                TransformErrorType.INVALID_INPUT,
                f"Empty url in row {row}",
            )
        else:
            url_response_text = await self._load_url(url)

        transformed_row = {
            self.output_columns["content_column"]: url_response_text,
        }
        return self._return_output_row(transformed_row)

    def params(self):
        return {
            "url_column": self.url_column,
            "output_columns": self.output_columns,
        }
