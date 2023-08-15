from autolabel.transforms.webpage_transform import WebpageTransform
import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_webpage_transform():
    # Initialize the transform class
    transform = WebpageTransform(
        output_columns={
            "content_column": "webpage_content",
            "metadata_column": "metadata",
        },
        url_column="url",
        cache=None,
    )

    # Create a mock row
    row = {"url": "https://en.wikipedia.org/wiki/Main_Page"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(["webpage_content", "metadata", "webpage_transform_applied_successfully"])
    assert isinstance(transformed_row["webpage_content"], str)
    assert isinstance(transformed_row["metadata"], dict)
    assert transformed_row["webpage_transform_applied_successfully"] == True
    assert len(transformed_row["webpage_content"]) > 0


@pytest.mark.asyncio
async def test_error_handling():
    # Initialize the transform class
    transform = WebpageTransform(
        output_columns={
            "content_column": "webpage_content",
        },
        url_column="url",
        cache=None,
    )

    # Create a mock row
    row = {"url": "bad_url"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(
        ["webpage_content", "webpage_transform_applied_successfully"]
    )
    assert transformed_row["webpage_content"] is None
    assert transformed_row["webpage_transform_applied_successfully"] == False
