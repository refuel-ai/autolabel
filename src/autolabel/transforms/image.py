from typing import Dict, Any

from autolabel.transforms.schema import TransformType
from autolabel.transforms import BaseTransform
from autolabel.cache import BaseCache


class ImageTransform(BaseTransform):
    """This class is used to extract text from images using OCR. The output columns dictionary for this class should include the keys 'content_column' and 'metadata_column'

    This transform supports the following image formats: PNG, JPEG, TIFF, JPEG 2000, GIF, WebP, BMP, and PNM
    """

    COLUMN_NAMES = [
        "content_column",
        "metadata_column",
    ]

    def __init__(
        self,
        cache: BaseCache,
        output_columns: Dict[str, Any],
        file_path_column: str,
        lang: str = None,
    ) -> None:
        super().__init__(cache, output_columns)
        self.file_path_column = file_path_column
        self.lang = lang

        try:
            from PIL import Image
            import pytesseract

            self.Image = Image
            self.pytesseract = pytesseract
            self.pytesseract.get_tesseract_version()
        except ImportError:
            raise ImportError(
                "pillow and pytesseract are required to use the image transform with ocr. Please install pillow and pytesseract with the following command: pip install pillow pytesseract"
            )
        except EnvironmentError:
            raise EnvironmentError(
                "The tesseract engine is required to use the image transform with ocr. Please see https://tesseract-ocr.github.io/tessdoc/Installation.html for installation instructions."
            )

    @staticmethod
    def name() -> str:
        return TransformType.IMAGE

    def get_image_metadata(self, file_path: str):
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
        except Exception as e:
            return {"error": str(e)}

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """This function transforms an image into text using OCR.

        Args:
            row (Dict[str, Any]): The row of data to be transformed.

        Returns:
            Dict[str, Any]: The dict of output columns.
        """
        content = self.pytesseract.image_to_string(
            row[self.file_path_column], lang=self.lang
        )
        metadata = self.get_image_metadata(row[self.file_path_column])
        transformed_row = {
            self.output_columns["content_column"]: content,
            self.output_columns["metadata_column"]: metadata,
        }
        return transformed_row

    def params(self) -> Dict[str, Any]:
        return {
            "output_columns": self.output_columns,
            "file_path_column": self.file_path_column,
            "lang": self.lang,
        }
