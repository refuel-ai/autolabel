import pytest

from autolabel.transforms.image import ImageTransform

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_image_transform(mocker):
    mocker.patch(
        "subprocess.check_output",
        return_value="5.3.2".encode("utf-8"),
    )
    mocker.patch(
        "pytesseract.pytesseract.run_and_get_output",
        return_value="This is a test",
    )

    # Initialize the ImageTransform class
    transform = ImageTransform(
        output_columns={
            "content_column": "content",
            "metadata_column": "metadata",
        },
        file_path_column="file_path",
        cache=None,
    )

    # Create a mock row
    row = {"file_path": "tests/assets/transforms/budget.png"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(
        ["content", "metadata", "content_error", "metadata_error"]
    )
    assert transformed_row["content_error"] is None
    assert transformed_row["content"] == "This is a test"
    assert isinstance(transformed_row["metadata"], dict)
    assert len(transformed_row["content"]) > 0
    metadata = transformed_row["metadata"]
    assert metadata["format"] == row["file_path"].split(".")[-1].upper()
    assert metadata["mode"] == "L"
    assert metadata["size"] == (1766, 2257)
    assert metadata["width"] == 1766
    assert metadata["height"] == 2257
    assert metadata["exif"] is None


@pytest.mark.asyncio
async def test_error_handling():
    # Initialize the PDFTransform class
    transform = ImageTransform(
        output_columns={
            "content_column": "content",
        },
        file_path_column="file_path",
        cache=None,
    )

    # Create a mock row
    row = {"file_path": "invalid_file.png"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(["content", "content_error"])
    assert (
        transformed_row["content"]
        == "tesseract is not installed or it's not in your PATH. See README file for more information."
    )
    assert (
        transformed_row["content_error"]
        == "tesseract is not installed or it's not in your PATH. See README file for more information."
    )
