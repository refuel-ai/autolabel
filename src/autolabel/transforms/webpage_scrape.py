from autolabel.transforms.schema import (
    TransformType,
    TransformError,
    TransformErrorType,
)
from autolabel.transforms import BaseTransform
from typing import Dict, Any
from urllib.parse import urlparse
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
MAX_KEEPALIVE_CONNECTIONS = 20
CONNECTION_TIMEOUT = 10
MAX_CONNECTIONS = 100
BACKOFF = 2
HEADERS = {}
HTML_PARSER = "html.parser"
SCRAPINGBEE_TIMEOUT = 60000  # in milliseconds
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


class WebpageScrape(BaseTransform):
    COLUMN_NAMES = [
        "content_column",
    ]

    def __init__(
        self,
        cache: BaseCache,
        output_columns: Dict[str, Any],
        url_column: str,
        timeout: int = 60,
        headers: Dict[str, str] = HEADERS,
        scrapingbee_api_key: str = None,
    ) -> None:
        super().__init__(cache, output_columns)
        self.url_column = url_column
        self.headers = headers
        self.api_key = scrapingbee_api_key
        self.html2text_transformer = Html2TextTransformer()
        self.max_retries = MAX_RETRIES
        self.scrapingbee_params = {
            "timeout": SCRAPINGBEE_TIMEOUT,
            "transparent_status_code": "True",
            "js_scenario": JS_SCENARIO,
        }
        try:
            from bs4 import BeautifulSoup
            import httpx

            if not headers.get("User-Agent"):
                from fake_useragent import UserAgent

                headers["User-Agent"] = UserAgent().random

            self.httpx = httpx
            self.timeout_time = timeout
            self.timeout = httpx.Timeout(connect=CONNECTION_TIMEOUT, timeout=timeout)
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
        return TransformType.WEBPAGE_SCRAPE

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
            response.raise_for_status()
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
            return await self._load_url_scrapingbee(url, retry_count=retry_count + 1)
        except Exception as e:
            logger.error(f"Error fetching content from URL: {url}. Exception: {e}")
            raise e

    async def _load_url_scrapingbee(self, url: str, retry_count=0) -> str:
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
            return await self._load_url_scrapingbee(url, retry_count=retry_count + 1)

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        url = row[self.url_column]
        url_response_data = {}
        if pd.isna(url) or url == self.NULL_TRANSFORM_TOKEN:
            raise TransformError(
                TransformErrorType.INVALID_INPUT,
                f"Empty url in row {row}",
            )
        else:
            if not urlparse(url).scheme:
                url = f"https://{url}"
            url_response_data = await self._load_url(url)

        transformed_row = {
            self.output_columns["content_column"]: url_response_data.get("content"),
        }

        return self._return_output_row(transformed_row)

    def params(self):
        return {
            "url_column": self.url_column,
            "output_columns": self.output_columns,
            "timeout": self.timeout_time,
            "scrapingbee_api_key": self.api_key,
        }

    def input_columns(self):
        return [self.url_column]
