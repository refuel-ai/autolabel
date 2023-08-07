from autolabel.schema import TransformType
from autolabel.transforms import BaseTransform
from typing import Dict, Any
import asyncio
import logging
import pandas as pd
import ssl

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
MAX_KEEPALIVE_CONNECTIONS = 20
MAX_CONNECTIONS = 100
BACKOFF = 2
HEADERS = {}
HTML_PARSER = "html.parser"


class WebpageTransform(BaseTransform):
    def __init__(
        self,
        url_column: str,
        output_columns: Dict[str, Any],
        timeout: int = 5,
        headers: Dict[str, str] = HEADERS,
    ) -> None:
        super().__init__(output_columns)
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
                "BeautifulSoup, httpx and fake_useragent are required to use the webpage transform. Please install them with the following command: pip install bs4 httpx fake_useragent"
            )

    def name(self) -> str:
        return TransformType.WEBPAGE_TRANSFORM

    @property
    def output_columns(self) -> Dict[str, Any]:
        COLUMN_NAMES = [
            "content_column",
            "content_in_bytes_column",
            "soup_column",
            "metadata_column",
        ]
        return {k: self._output_columns.get(k, k) for k in COLUMN_NAMES}

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
            return {}

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
            return {}
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
            return {}

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        url = row[self.url_column]
        url_response_data = {}
        if pd.isna(url):
            logger.warning(f"Empty url in row {row}")
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

        return transformed_row
