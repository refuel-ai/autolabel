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

from autolabel.cache import BaseCache

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
MAX_KEEPALIVE_CONNECTIONS = 20
MAX_CONNECTIONS = 100
BACKOFF = 2
HEADERS = {}
HTML_PARSER = "html.parser"


class WebpageTransform(BaseTransform):
    COLUMN_NAMES = [
        "content_column",
        "content_in_bytes_column",
        "soup_column",
        "metadata_column",
    ]

    def __init__(
        self,
        cache: BaseCache,
        output_columns: Dict[str, Any],
        url_column: str,
        timeout: int = 60,
        headers: Dict[str, str] = HEADERS,
    ) -> None:
        super().__init__(cache, output_columns)
        self.url_column = url_column
        self.headers = headers
        self.max_retries = MAX_RETRIES
        try:
            from bs4 import BeautifulSoup
            import httpx

            if not headers.get("User-Agent"):
                from fake_useragent import UserAgent

                headers["User-Agent"] = UserAgent().random

            self.httpx = httpx
            self.timeout_time = timeout
            self.timeout = httpx.Timeout(timeout)
            limits = httpx.Limits(
                max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
                max_connections=MAX_CONNECTIONS,
                keepalive_expiry=timeout,
            )
            self.client = httpx.AsyncClient(
                timeout=self.timeout, limits=limits, follow_redirects=True
            )
            self.client_with_no_verify = httpx.AsyncClient(
                timeout=self.timeout, limits=limits, follow_redirects=True, verify=False
            )
            self.beautiful_soup = BeautifulSoup
        except ImportError:
            raise ImportError(
                "BeautifulSoup, httpx and fake_useragent are required to use the webpage transform. Please install them with the following command: pip install beautifulsoup4 httpx fake_useragent"
            )

    def name(self) -> str:
        return TransformType.WEBPAGE_TRANSFORM

    def _load_metadata(self, url, soup) -> Dict[str, Any]:
        metadata = {"url": url}
        if soup.find("title"):
            metadata["title"] = soup.find("title").get_text()
        for meta in soup.find_all("meta"):
            if meta.get("name") and meta.get("content"):
                metadata[meta.get("name")] = meta.get("content")
            elif meta.get("property") and meta.get("content"):
                metadata[meta.get("property")] = meta.get("content")
        return metadata

    async def _load_url(
        self, url: str, verify=True, headers=HEADERS, retry_count=0
    ) -> Dict[str, Any]:
        if retry_count >= self.max_retries:
            logger.warning(f"Max retries reached for URL: {url}")
            raise TransformError(
                TransformErrorType.MAX_RETRIES_REACHED, "Max retries reached"
            )

        try:
            client = self.client
            if not verify:
                client = self.client_with_no_verify
            response = await client.get(url, headers=headers)

            # TODO: Add support for other parsers
            content_bytes = response.content
            soup = self.beautiful_soup(content_bytes, HTML_PARSER)
            return {
                "content": soup.get_text(),
                "content_bytes": content_bytes,
                "soup": soup,
                "metadata": self._load_metadata(url, soup),
            }
        except self.httpx.ConnectTimeout as e:
            logger.error(f"Timeout when fetching content from URL: {url}")
            raise TransformError(
                TransformErrorType.TRANSFORM_TIMEOUT,
                "Timeout when fetching content from URL",
            )
        except ssl.SSLCertVerificationError as e:
            logger.warning(
                f"SSL verification error when fetching content from URL: {url}, retrying with verify=False"
            )
            await asyncio.sleep(BACKOFF**retry_count)
            return await self._load_url(
                url, verify=False, headers=headers, retry_count=retry_count + 1
            )
        except Exception as e:
            logger.error(f"Error fetching content from URL: {url}. Exception: {e}")
            raise e

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        url = row[self.url_column]
        url_response_data = {}
        if pd.isna(url) or url == self.NULL_TRANSFORM_TOKEN:
            raise TransformError(
                TransformErrorType.INVALID_INPUT,
                f"Empty url in row {row}",
            )
        else:
            url_response_data = await self._load_url(url)

        transformed_row = {
            self.output_columns["content_column"]: url_response_data.get("content"),
            self.output_columns["content_in_bytes_column"]: url_response_data.get(
                "content_bytes"
            ),
            self.output_columns["soup_column"]: url_response_data.get("soup"),
            self.output_columns["metadata_column"]: url_response_data.get("metadata"),
        }

        return self._return_output_row(transformed_row)

    def params(self):
        return {
            "url_column": self.url_column,
            "output_columns": self.output_columns,
            "timeout": self.timeout_time,
        }
