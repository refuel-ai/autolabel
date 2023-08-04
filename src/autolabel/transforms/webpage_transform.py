from autolabel.schema import TransformType
from autolabel.transforms import BaseTransform
from typing import List, Dict, Any, Tuple
import asyncio
import logging
import pandas as pd

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
        output_columns: List[str],
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
    ) -> Tuple[str, Dict[str, Any]]:
        if retry_count >= self.max_retries:
            logger.warning(f"Max retries reached for URL: {url}")
            return None, None
        try:
            client = self.client
            if not verify:
                client = self.client_with_no_verify
            response = await client.get(url, headers=headers)
            # TODO: Add support for other parsers
            soup = self.beautiful_soup(response.text, HTML_PARSER)
            return soup.get_text(), self._load_metadata(url, soup)
        except Exception as e:
            if "RemoteDisconnected" in str(e):
                await asyncio.sleep(BACKOFF**retry_count)
                return await self._load_url(
                    url, verify=verify, headers={}, retry_count=retry_count + 1
                )
            elif "CERTIFICATE_VERIFY_FAILED" in str(e):
                await asyncio.sleep(BACKOFF**retry_count)
                return await self._load_url(
                    url, verify=False, headers=headers, retry_count=retry_count + 1
                )
            elif isinstance(e, self.httpx.ConnectTimeout):
                logger.error(f"Timeout when fetching content from URL: {url}")
                return None, None
            else:
                logger.error(f"Error fetching content from URL: {url}. Exception: {e}")
                return None, None

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        url = row[self.url_column]
        content, metadata = None, None
        if pd.isna(url):
            logger.warning(f"Empty url in row {row}")
        else:
            content, metadata = await self._load_url(url)

        return {
            column: value
            for column, value in zip(self.output_columns, [content, metadata])
        }
