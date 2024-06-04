import asyncio
from collections import defaultdict
import json
from urllib.parse import urlparse
from autolabel.cache import BaseCache
from autolabel.transforms import BaseTransform
from langchain_community.utilities import GoogleSerperAPIWrapper
from typing import Dict, Any, List
import logging
import pandas as pd
import ssl

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
        base_url: str,
        request_columns: List[str],
        headers: Dict[str, str] = HEADERS,
        timeout: int = 60,
    ) -> None:
        super().__init__(cache, output_columns)
        self.request_columns = request_columns
        if not urlparse(base_url).scheme:
            base_url = f"https://{base_url}"
        self.base_url = base_url
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
                timeout=self.timeout, limits=limits, follow_redirects=True
            )
            self.client_with_no_verify = httpx.AsyncClient(
                timeout=self.timeout, limits=limits, follow_redirects=True, verify=False
            )
        except ImportError:
            raise ImportError(
                "httpx and fake_useragent are required to use the custom API transform. Please install them with the following command: pip install httpx fake_useragent"
            )

    def name(self) -> str:
        return TransformType.CUSTOM_API

    async def _get_result(
        self, url: str, params: Dict, verify=True, headers=HEADERS, retry_count=0
    ) -> Dict[str, Any]:
        if retry_count >= self.max_retries:
            logger.warning(f"Max retries reached for URL: {url}")
            raise TransformError(
                TransformErrorType.MAX_RETRIES_REACHED,
                f"Max retries reached for URL: {url}",
            )

        try:
            client = self.client
            if not verify:
                client = self.client_with_no_verify
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.text
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
            return await self._get_result(
                url, verify=False, headers=headers, retry_count=retry_count + 1
            )
        except Exception as e:
            logger.error(f"Error fetching content from URL: {url}. Exception: {e}")
            raise e

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        params = {}
        for col in self.request_columns:
            if col not in row:
                logger.error(
                    f"Missing request column: {col} in row {row}",
                )
            else:
                params[col] = row.get(col)
        result = await self._get_result(self.base_url, params)
        transformed_row = {self.output_columns["result"]: result}
        return self._return_output_row(transformed_row)

    def params(self):
        return {
            "output_columns": self.output_columns,
            "base_url": self.base_url,
            "request_columns": self.request_columns,
        }

    def input_columns(self) -> List[str]:
        return self.request_columns
