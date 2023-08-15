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
    assert set(transformed_row.keys()) == set(["webpage_content", "metadata"])
    assert isinstance(transformed_row["webpage_content"], str)
    assert isinstance(transformed_row["metadata"], dict)
    assert len(transformed_row["webpage_content"]) > 0
