"""OCR transform for extracting text from documents."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, ClassVar

import boto3
from PIL import Image

from autolabel.cache import BaseCache
from autolabel.transforms import BaseTransform
from autolabel.transforms.schema import (
    TransformError,
    TransformErrorType,
    TransformType,
)


class OCRTransform(BaseTransform):
    """Extract text from documents using OCR.

    The output columns dictionary for this class should include the keys 'content_column'
    and 'metadata_column'.

    This transform supports the following image formats: PDF, PNG, JPEG, TIFF, JPEG 2000,
    GIF, WebP, BMP, and PNM.
    """

    COLUMN_NAMES: ClassVar[list[str]] = ["content_column"]

    def __init__(
        self,
        cache: BaseCache,
        output_columns: dict[str, Any],
        url_column: str,
        lang: str | None = None,
    ) -> None:
        """Initialize OCRTransform.

        Args:
            cache: Cache instance to use
            output_columns: Dictionary mapping output column names
            url_column: Column containing document URLs/paths
            lang: Optional language for OCR
            custom_image_ocr: Optional custom OCR function

        """
        super().__init__(cache, output_columns)
        self.url_column = url_column
        self.lang = lang
        try:
            import pytesseract
            from pdf2image import convert_from_path

            self.pytesseract = pytesseract
            self.convert_from_path = convert_from_path
            self.pytesseract.get_tesseract_version()

            self.session = boto3.Session()
            self.s3_connection = self.session.resource("s3")
            self.client = self.session.client("textract")

        except ImportError:
            msg = (
                "pillow, pytesseract, and pdf2image are required to use ocr"
                "Please install with: pip install pillow pytesseract pdf2image"
            )
            raise ImportError(msg) from None
        except OSError:
            msg = (
                "The tesseract engine is required to use the ocr transform. "
                "Please see https://tesseract-ocr.github.io/tessdoc/Installation.html "
                "for installation instructions."
            )
            raise OSError(msg) from None

    @staticmethod
    def name() -> str:
        """Get transform name.

        Returns:
            Transform type name

        """
        return TransformType.OCR

    def default_ocr_processor(
        self,
        image_or_image_path: Image.Image | str,
        lang: str | None = None,
    ) -> str:
        """Extract text from image using OCR.

        Args:
            image_or_image_path: PIL Image or path to image file
            lang: Optional language for OCR

        Returns:
            Extracted text

        """
        image = image_or_image_path
        if isinstance(image_or_image_path, str):
            image = Image.open(image_or_image_path)

        if not isinstance(image, Image.Image):
            raise TransformError(
                TransformErrorType.TRANSFORM_ERROR,
                "Invalid image type",
            )
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        response = self.client.detect_document_text(Document={"Bytes": img_bytes})
        blocks = response["Blocks"]
        return "\n".join([block.get("Text", "") for block in blocks])

    def download_file(self, file_location: str) -> str:
        """Download file from URL to temporary location.

        Args:
            file_location: URL or path of file to download

        Returns:
            Path to downloaded temporary file

        """
        import tempfile

        import requests

        ext = Path(file_location).suffix
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            response = requests.get(file_location, timeout=30)
            response.raise_for_status()

            with Path(temp_file.name).open("wb") as f:
                f.write(response.content)

        return temp_file.name

    async def _apply(self, row: dict[str, Any]) -> dict[str, Any]:
        """Transform document into text using OCR.

        Args:
            row: Row of data to transform

        Returns:
            Dict of output columns

        Raises:
            TransformError: If file download fails

        """
        curr_file_location = row[self.url_column]
        try:
            curr_file_path = self.download_file(curr_file_location)
        except Exception as exc:
            raise TransformError(
                TransformErrorType.TRANSFORM_ERROR,
                f"Error downloading file: {exc}",
            ) from exc

        ocr_output = []
        if curr_file_path.endswith(".pdf"):
            pages = self.convert_from_path(curr_file_path)
            ocr_output = [
                self.default_ocr_processor(page, lang=self.lang) for page in pages
            ]
        else:
            ocr_output = [self.default_ocr_processor(curr_file_path, lang=self.lang)]

        transformed_row = {
            self.output_columns["content_column"]: "\n\n".join(ocr_output),
        }
        return self._return_output_row(transformed_row)

    def params(self) -> dict[str, Any]:
        """Get transform parameters.

        Returns:
            Dict of parameters

        """
        return {
            "output_columns": self.output_columns,
            "url_column": self.url_column,
            "lang": self.lang,
        }

    def input_columns(self) -> list[str]:
        """Get required input columns.

        Returns:
            List of input column names

        """
        return [self.url_column]
