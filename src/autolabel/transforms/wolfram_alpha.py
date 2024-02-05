from autolabel.transforms.schema import (
    TransformType,
    TransformError,
    TransformErrorType,
)
from autolabel.transforms import BaseTransform
from typing import Dict, Any, List
import asyncio
import logging
import pandas as pd
import ssl

from autolabel.cache import BaseCache

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
MAX_KEEPALIVE_CONNECTIONS = 20
MAX_CONNECTIONS = 100
BACKOFF = 2
HEADERS = {}
API_BASE_URL = "https://www.wolframalpha.com/api/v1/llm-api"


class WolframAlpha(BaseTransform):
    COLUMN_NAMES = [
        "result",
    ]

    def __init__(
        self,
        cache: BaseCache,
        output_columns: Dict[str, Any],
        query_columns: List[str],
        query_template: str,
        wolfram_app_id: str,
        wolfram_args: Dict[str, Any] = {},
        timeout: int = 5,
    ) -> None:
        super().__init__(cache, output_columns)
        self.max_retries = MAX_RETRIES
        self.query_columns = query_columns
        self.query_template = query_template
        self.wolfram_app_id = wolfram_app_id
        self.wolfram_args = wolfram_args
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.wolfram_app_id}",
        }
        try:
            import httpx

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
        except ImportError:
            raise ImportError(
                "httpx is required to use the wolfram alpha transform. Please install them with the following command: pip install httpx"
            )

    def name(self) -> str:
        return TransformType.WOLFRAM_ALPHA_API

    async def _get_result(
        self, query: str, verify=True, headers=HEADERS, retry_count=0
    ) -> Dict[str, Any]:
        if retry_count >= self.max_retries:
            logger.warning(f"Max retries reached for query: {query}")
            raise TransformError(
                TransformErrorType.MAX_RETRIES_REACHED, "Max retries reached"
            )

        try:
            client = self.client
            if not verify:
                client = self.client_with_no_verify
            params = self.wolfram_args
            params["input"] = query
            response = await client.get(API_BASE_URL, headers=headers, params=params)
            if response.status_code != 200:
                logger.debug(
                    f"Error fetching content. Status code: {response.status_code}"
                )
                raise TransformError(
                    TransformErrorType.TRANSFORM_ERROR,
                    f"Error fetching content. Status code: {response.status_code}",
                )
            return {
                "result": response.text,
            }
        except self.httpx.ConnectTimeout as e:
            logger.error(f"Timeout when fetching content")
            raise TransformError(
                TransformErrorType.TRANSFORM_TIMEOUT,
                "Timeout when fetching content",
            )
        except ssl.SSLCertVerificationError as e:
            logger.warning(
                f"SSL verification error when fetching content, retrying with verify=False"
            )
            await asyncio.sleep(BACKOFF**retry_count)
            return await self._get_result(
                query, verify=False, headers=headers, retry_count=retry_count + 1
            )
        except Exception as e:
            logger.error(f"Error fetching content. Exception: {e}")
            raise e

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        for col in self.query_columns:
            if col not in row:
                raise TransformError(
                    TransformErrorType.INVALID_INPUT,
                    f"Missing query column: {col} in row {row}",
                )
        query = self.query_template.format(**row)
        result = {}
        if pd.isna(query) or query == self.NULL_TRANSFORM_TOKEN:
            raise TransformError(
                TransformErrorType.INVALID_INPUT,
                f"Empty query in row {row}",
            )
        else:
            result = await self._get_result(query, headers=self.headers)

        transformed_row = {self.output_columns["result"]: result}

        return self._return_output_row(transformed_row)

    def params(self):
        return {
            "query_columns": self.query_columns,
            "query_template": self.query_template,
            "output_columns": self.output_columns,
            "wolfram_app_id": self.wolfram_app_id,
            "wolfram_args": self.wolfram_args,
        }
