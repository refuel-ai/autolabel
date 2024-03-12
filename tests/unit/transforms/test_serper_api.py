import json
from unittest.mock import Mock

import pytest

from autolabel.transforms.serper_api import RefuelSerperAPIWrapper, SerperApi

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_webpage_transform():
    serper_api_wrapper_mock = Mock(spec=RefuelSerperAPIWrapper)
    serper_api_wrapper_mock.arun.return_value = {
        "knowledge_graph": json.dumps(
            {"title": "Joe Biden", "type": "46th U.S. President"}
        ),
        "organic_results": json.dumps(
            [
                {
                    "position": 1,
                    "title": "Presidents",
                    "link": "https://www.whitehouse.gov/about-the-white-house/presidents/",
                }
            ]
        ),
    }
    # Initialize the transform class
    transform = SerperApi(
        output_columns={
            "knowledge_graph_results": "knowledge_graph_results",
            "organic_search_results": "organic_search_results",
        },
        query_columns=["query"],
        query_template="{query}",
        serper_api_key="test_key",
        cache=None,
    )

    transform.serper_api_wrapper = serper_api_wrapper_mock

    # Create a mock row
    row = {"query": "Who is the president of the United States?"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(
        ["knowledge_graph_results", "organic_search_results"]
    )
    assert (
        json.loads(transformed_row["knowledge_graph_results"])["title"] == "Joe Biden"
    )


@pytest.mark.asyncio
async def test_null_query():
    serper_api_wrapper_mock = Mock(spec=RefuelSerperAPIWrapper)
    serper_api_wrapper_mock.arun.return_value = "Test Response"
    # Initialize the transform class
    transform = SerperApi(
        output_columns={
            "knowledge_graph_results": "knowledge_graph_results",
            "organic_search_results": "organic_search_results",
        },
        query_columns=["query"],
        query_template="{query}",
        serper_api_key="test_key",
        cache=None,
    )

    transform.serper_api_wrapper = serper_api_wrapper_mock

    # Create a mock row
    row = {"query": transform.NULL_TRANSFORM_TOKEN}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(
        ["knowledge_graph_results", "web_search_error", "organic_search_results"]
    )
    assert transformed_row["knowledge_graph_results"] == "NO_TRANSFORM"
    assert (
        transformed_row["web_search_error"]
        == "INVALID_INPUT: Empty query in row {'query': 'NO_TRANSFORM'}"
    )
