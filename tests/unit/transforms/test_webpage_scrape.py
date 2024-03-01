from unittest.mock import Mock
from autolabel.transforms.webpage_scrape import WebpageScrape
import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_webpage_scrape():
    # Initialize the transform class
    webpage_scrape_mock = Mock(spec=WebpageScrape)
    webpage_scrape_mock.apply.return_value = {
        "webpage_content": "test_content",
    }
    transform = WebpageScrape(
        output_columns={
            "content_column": "webpage_content",
        },
        url_column="url",
        scrapingbee_api_key="test_key",
        cache=None,
    )

    # Create a mock row
    row = {"url": "https://en.wikipedia.org/wiki/Main_Page"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(
        ["webpage_content", "webpage_scrape_error"]
    )
    assert isinstance(transformed_row["webpage_content"], str)
    assert len(transformed_row["webpage_content"]) > 0


@pytest.mark.asyncio
async def test_error_handling():
    # Initialize the transform class
    transform = WebpageScrape(
        output_columns={
            "content_column": "webpage_content",
        },
        url_column="url",
        scrapingbee_api_key="test_key",
        cache=None,
    )

    # Create a mock row
    row = {"url": transform.NULL_TRANSFORM_TOKEN}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(
        ["webpage_content", "webpage_scrape_error"]
    )
    assert transformed_row["webpage_content"] == "NO_TRANSFORM"
    assert (
        transformed_row["webpage_scrape_error"]
        == "INVALID_INPUT: Empty url in row {'url': 'NO_TRANSFORM'}"
    )
