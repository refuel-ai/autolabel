import json
from unittest.mock import Mock

import pytest

from autolabel.transforms.serp_api import RefuelSerpAPIWrapper, SerpApi

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_webpage_transform():
    serp_api_wrapper_mock = Mock(spec=RefuelSerpAPIWrapper)
    serp_api_wrapper_mock.arun.return_value = {
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
    transform = SerpApi(
        output_columns={
            "knowledge_graph_results": "knowledge_graph_results",
            "organic_search_results": "organic_search_results",
        },
        query_columns=["query"],
        query_template="{query}",
        serp_api_key="test_key",
        cache=None,
    )

    transform.serp_api_wrapper = serp_api_wrapper_mock

    # Create a mock row
    row = {"query": "Who is the president of the United States?"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(
        [
            "knowledge_graph_results",
            "organic_search_results",
            "knowledge_graph_results_error",
            "organic_search_results_error",
        ]
    )
    assert (
        json.loads(transformed_row["knowledge_graph_results"])["title"] == "Joe Biden"
    )


@pytest.mark.asyncio
async def test_error_handling():
    # Initialize the transform class
    transform = SerpApi(
        output_columns={
            "knowledge_graph_results": "knowledge_graph_results",
            "organic_search_results": "organic_search_results",
        },
        query_columns=["query"],
        query_template="{query}",
        serp_api_key="test_key",
        cache=None,
    )

    # Create a mock row
    row = {"query": "Who is the president of the United States?"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(
        [
            "knowledge_graph_results",
            "knowledge_graph_results_error",
            "organic_search_results",
            "organic_search_results_error",
        ]
    )
    assert (
        "Error while making request with query"
        in transformed_row["knowledge_graph_results"]
    )
    assert (
        "Error while making request with query"
        in transformed_row["knowledge_graph_results_error"]
    )


@pytest.mark.asyncio
async def test_null_query():
    serp_api_wrapper_mock = Mock(spec=RefuelSerpAPIWrapper)
    serp_api_wrapper_mock.arun.return_value = "Test Response"
    # Initialize the transform class
    transform = SerpApi(
        output_columns={
            "knowledge_graph_results": "knowledge_graph_results",
            "organic_search_results": "organic_search_results",
        },
        query_columns=["query"],
        query_template="{query}",
        serp_api_key="test_key",
        cache=None,
    )

    transform.serp_api_wrapper = serp_api_wrapper_mock

    # Create a mock row
    row = {"query": transform.NULL_TRANSFORM_TOKEN}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(
        [
            "knowledge_graph_results",
            "knowledge_graph_results_error",
            "organic_search_results",
            "organic_search_results_error",
        ]
    )
    assert (
        transformed_row["knowledge_graph_results"]
        == "INVALID_INPUT: Empty query in row {'query': 'NO_TRANSFORM'}"
    )
    assert (
        transformed_row["knowledge_graph_results_error"]
        == "INVALID_INPUT: Empty query in row {'query': 'NO_TRANSFORM'}"
    )
