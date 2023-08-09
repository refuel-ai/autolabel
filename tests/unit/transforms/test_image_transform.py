from autolabel.transforms.image import ImageTransform
import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_image_transform():
    # Initialize the ImageTransform class
    transform = ImageTransform(
        output_columns={
            "content_column": "content",
            "metadata_column": "metadata",
        },
        file_path_column="file_path",
    )

    # Create a mock row
    row = {"file_path": "tests/assets/transforms/budget.png"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(["content", "metadata"])
    assert isinstance(transformed_row["content"], str)
    assert isinstance(transformed_row["metadata"], dict)
    assert len(transformed_row["content"]) > 0
    metadata = transformed_row["metadata"]
    assert metadata["format"] == row["file_path"].split(".")[-1].upper()
    assert metadata["mode"] == "L"
    assert metadata["size"] == (1766, 2257)
    assert metadata["width"] == 1766
    assert metadata["height"] == 2257
    assert metadata["exif"] is None
