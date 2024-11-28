"""Extract text from images using OCR."""

from __future__ import annotations

from typing import Any, ClassVar

from autolabel.cache import BaseCache
from autolabel.transforms import BaseTransform
from autolabel.transforms.schema import TransformType


class ImageTransform(BaseTransform):

    """
    Extract text from images using OCR.

    The output columns dictionary for this class should include the keys 'content_column'
    and 'metadata_column'.

    This transform supports the following image formats: PNG, JPEG, TIFF, JPEG 2000, GIF,
    WebP, BMP, and PNM.
    """

    COLUMN_NAMES: ClassVar[list[str]] = [
        "content_column",
        "metadata_column",
    ]

    def __init__(
        self,
        cache: BaseCache,
        output_columns: dict[str, Any],
        file_path_column: str,
        lang: str | None = None,
    ) -> None:
        """
        Initialize the ImageTransform.

        Args:
            cache: Cache instance to use
            output_columns: Dictionary mapping output column names
            file_path_column: Column containing image file paths
            lang: Optional language for OCR

        """
        super().__init__(cache, output_columns)
        self.file_path_column = file_path_column
        self.lang = lang

        try:
            import pytesseract
            from PIL import Image

            self.Image = Image
            self.pytesseract = pytesseract
            self.pytesseract.get_tesseract_version()
        except ImportError:
            msg = (
                "pillow and pytesseract required to use the image transform with ocr"
                "Please install pillow and pytesseract with the following command: "
                "pip install pillow pytesseract"
            )
            raise ImportError(msg) from None
        except OSError:
            msg = (
                "The tesseract engine is required to use the image transform with ocr. "
                "Please see https://tesseract-ocr.github.io/tessdoc/Installation.html "
                "for installation instructions."
            )
            raise OSError(msg) from None

    @staticmethod
    def name() -> str:
        """
        Get transform name.

        Returns:
            Transform type name

        """
        return TransformType.IMAGE

    def get_image_metadata(self, file_path: str) -> dict[str, Any]:
        """
        Get metadata from image file.

        Args:
            file_path: Path to image file

        Returns:
            Dictionary of image metadata

        """
        try:
            image = self.Image.open(file_path)
            metadata = {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.width,
                "height": image.height,
                "exif": image._getexif(),  # Exif metadata
            }
            return metadata
        except Exception as exc:
            return {"error": str(exc)}

    async def _apply(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Transform an image into text using OCR.

        Args:
            row: The row of data to be transformed

        Returns:
            Dictionary of output columns

        """
        content = self.pytesseract.image_to_string(
            row[self.file_path_column],
            lang=self.lang,
        )
        metadata = self.get_image_metadata(row[self.file_path_column])
        transformed_row = {
            self.output_columns["content_column"]: content,
            self.output_columns["metadata_column"]: metadata,
        }
        return transformed_row

    def params(self) -> dict[str, Any]:
        """
        Get transform parameters.

        Returns:
            Dictionary of parameters

        """
        return {
            "output_columns": self.output_columns,
            "file_path_column": self.file_path_column,
            "lang": self.lang,
        }

    def input_columns(self) -> list[str]:
        """
        Get required input columns.

        Returns:
            List of input column names

        """
        return [self.file_path_column]
