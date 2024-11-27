import asyncio
import logging
import ssl
from collections import defaultdict
from typing import Any, Dict, List
from urllib.parse import urlparse

from autolabel.cache import BaseCache
from autolabel.transforms import BaseTransform
from autolabel.transforms.schema import (
    TransformError,
    TransformErrorType,
    TransformType,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
MAX_KEEPALIVE_CONNECTIONS = 20
CONNECTION_TIMEOUT = 10
MAX_CONNECTIONS = 100
BACKOFF = 2
HEADERS = {}


class CustomApi(BaseTransform):
    COLUMN_NAMES = ["result"]

    def __init__(
        self,
        cache: BaseCache,
        output_columns: Dict[str, Any],
        api_url: str,
        request_columns: List[str],
        headers: Dict[str, str] = HEADERS,
        timeout: int = 60,
    ) -> None:
        super().__init__(cache, output_columns)
        self.request_columns = request_columns
        if not urlparse(api_url).scheme:
            api_url = f"https://{api_url}"
        self.api_url = api_url
        self.headers = headers
        self.max_retries = MAX_RETRIES
        try:
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
                timeout=self.timeout, limits=limits, follow_redirects=True,
            )
            self.client_with_no_verify = httpx.AsyncClient(
                timeout=self.timeout, limits=limits, follow_redirects=True, verify=False,
            )
        except ImportError:
            raise ImportError(
                "httpx and fake_useragent are required to use the custom API transform. Please install them with the following command: pip install httpx fake_useragent",
            )

    def name(self) -> str:
        return TransformType.CUSTOM_API

    async def _get_result(
        self, url: str, params: Dict = {}, verify=True, headers=HEADERS, retry_count=0,
    ) -> Dict[str, Any]:
        if retry_count >= self.max_retries:
            logger.warning(f"Max retries reached for URL: {url}")
            raise TransformError(
                TransformErrorType.MAX_RETRIES_REACHED,
                f"Max retries reached for URL: {url}",
            )
        request_url = url
        try:
            client = self.client
            if not verify:
                client = self.client_with_no_verify
            response = await client.get(url, headers=headers, params=params)
            request_url = response.url
            response.raise_for_status()
            return response.text
        except self.httpx.ConnectTimeout:
            logger.error(f"Timeout when making request to URL: {request_url}")
            raise TransformError(
                TransformErrorType.TRANSFORM_TIMEOUT,
                f"Timeout when making request to URL: {request_url}",
            )
        except ssl.SSLCertVerificationError:
            logger.warning(
                f"SSL verification error when making request to URL: {request_url}, retrying with verify=False",
            )
            await asyncio.sleep(BACKOFF**retry_count)
            return await self._get_result(
                url, verify=False, headers=headers, retry_count=retry_count + 1,
            )
        except Exception as e:
            logger.error(
                f"Error when making request to URL: {request_url}. Exception: {e!s}",
            )
            raise TransformError(
                TransformErrorType.TRANSFORM_API_ERROR,
                f"Error when making request to URL: {request_url}",
            )

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        for col in self.request_columns:
            if col not in row:
                logger.warning(
                    f"Missing request column: {col} in row {row}",
                )
        url = self.api_url.format_map(defaultdict(str, row))
        result = await self._get_result(url)
        transformed_row = {self.output_columns["result"]: result}
        return self._return_output_row(transformed_row)

    def params(self):
        return {
            "output_columns": self.output_columns,
            "api_url": self.api_url,
            "request_columns": self.request_columns,
        }

    def input_columns(self) -> List[str]:
        return self.request_columns
