from autolabel.transforms.serp_api import SerpApi
from unittest.mock import Mock
from langchain.utilities import SerpAPIWrapper
import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_webpage_transform():
    serp_api_wrapper_mock = Mock(spec=SerpAPIWrapper)
    serp_api_wrapper_mock.arun.return_value = "Joe Biden"
    # Initialize the transform class
    transform = SerpApi(
        output_columns={
            "result_column": "search_result",
        },
        query_column="query",
        serp_api_key="test_key",
        cache=None,
    )

    transform.serp_api_wrapper = serp_api_wrapper_mock

    # Create a mock row
    row = {"query": "Who is the president of the United States?"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(["search_result"])
    assert transformed_row["search_result"] == "Joe Biden"


@pytest.mark.asyncio
async def test_error_handling():
    # Initialize the transform class
    transform = SerpApi(
        output_columns={
            "result_column": "search_result",
        },
        query_column="query",
        serp_api_key="test_key",
        cache=None,
    )

    # Create a mock row
    row = {"query": "Who is the president of the United States?"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(["search_result", "serp_api_error"])
    assert transformed_row["search_result"] == "NO_TRANSFORM"
    assert "Invalid API key" in transformed_row["serp_api_error"]


@pytest.mark.asyncio
async def test_null_query():
    serp_api_wrapper_mock = Mock(spec=SerpAPIWrapper)
    serp_api_wrapper_mock.arun.return_value = "Test Response"
    # Initialize the transform class
    transform = SerpApi(
        output_columns={
            "result_column": "search_result",
        },
        query_column="query",
        serp_api_key="test_key",
        cache=None,
    )

    transform.serp_api_wrapper = serp_api_wrapper_mock

    # Create a mock row
    row = {"query": transform.NULL_TRANSFORM_TOKEN}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(["search_result", "serp_api_error"])
    assert transformed_row["search_result"] == "NO_TRANSFORM"
    assert (
        transformed_row["serp_api_error"]
        == "INVALID_INPUT: Empty query in row {'query': 'NO_TRANSFORM'}"
    )
