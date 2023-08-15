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
