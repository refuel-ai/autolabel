from autolabel.transforms.webpage_scrape import WebpageScrape
import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_webpage_scrape():
    # Initialize the transform class
    transform = WebpageScrape(
        output_columns={
            "content_column": "webpage_content",
            "metadata_column": "metadata",
        },
        url_column="url",
        cache=None,
    )

    # Create a mock row
    row = {"url": "en.wikipedia.org/wiki/Main_Page"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(["webpage_content", "metadata"])
    assert isinstance(transformed_row["webpage_content"], str)
    assert isinstance(transformed_row["metadata"], dict)
    assert len(transformed_row["webpage_content"]) > 0


@pytest.mark.asyncio
async def test_empty_url():
    # Initialize the transform class
    transform = WebpageScrape(
        output_columns={
            "content_column": "webpage_content",
        },
        url_column="url",
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


@pytest.mark.asyncio
async def test_unreachable_url():
    # Initialize the transform class
    transform = WebpageScrape(
        output_columns={
            "content_column": "webpage_content",
        },
        url_column="url",
        cache=None,
    )

    # Create a mock row
    row = {"url": "http://portal.net.kp/"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(
        ["webpage_content", "webpage_scrape_error"]
    )
    assert transformed_row["webpage_content"] == "NO_TRANSFORM"
    assert (
        transformed_row["webpage_scrape_error"]
        == "TRANSFORM_TIMEOUT: Timeout when fetching content from URL"
    )
