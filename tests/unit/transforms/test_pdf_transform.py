from autolabel.transforms.pdf import PDFTransform
import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_pdf_transform():
    # Initialize the PDFTransform class
    transform = PDFTransform(
        output_columns={
            "content_column": "content",
            "metadata_column": "metadata",
        },
        file_path_column="file_path",
        cache=None,
    )

    # Create a mock row
    row = {"file_path": "tests/assets/transforms/Resume.pdf"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(["content", "metadata"])
    assert isinstance(transformed_row["content"], str)
    assert isinstance(transformed_row["metadata"], dict)
    assert len(transformed_row["content"]) > 0
    assert transformed_row["metadata"]["num_pages"] == 1


@pytest.mark.asyncio
async def test_pdf_transform_ocr(mocker):
    mocker.patch(
        "subprocess.check_output",
        return_value="5.3.2".encode("utf-8"),
    )
    mocker.patch(
        "autolabel.transforms.pdf.PDFTransform.get_page_texts",
        return_value=["This is a test"],
    )
    transform = PDFTransform(
        output_columns={
            "content_column": "content",
            "metadata_column": "metadata",
        },
        file_path_column="file_path",
        ocr_enabled=True,
        cache=None,
    )

    # Create a mock row
    row = {"file_path": "tests/assets/transforms/Resume.pdf"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(["content", "metadata"])
    assert transformed_row["content"] == "Page 1: This is a test"
    assert isinstance(transformed_row["metadata"], dict)
    assert len(transformed_row["content"]) > 0
    assert transformed_row["metadata"]["num_pages"] == 1


async def test_error_handling():
    # Initialize the PDFTransform class
    transform = PDFTransform(
        output_columns={
            "content_column": "content",
            "metadata_column": "metadata",
        },
        file_path_column="file_path",
        cache=None,
    )

    # Create a mock row
    row = {"file_path": "invalid_file.pdf"}
    # Transform the row
    transformed_row = await transform.apply(row)
    # Check the output
    assert set(transformed_row.keys()) == set(["content", "pdf_error"])
    assert transformed_row["content"] == "NO_TRANSFORM"
    assert (
        transformed_row["pdf_error"]
        == "File path invalid_file.pdf is not a valid file or url"
    )
