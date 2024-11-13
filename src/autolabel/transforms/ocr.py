from typing import Dict, Any, List

from autolabel.transforms.schema import TransformType
from autolabel.transforms import BaseTransform
from autolabel.cache import BaseCache

from autolabel.transforms.schema import TransformError, TransformErrorType


class OCRTransform(BaseTransform):
    """This class is used to extract text from any document using OCR. The output columns dictionary for this class should include the keys 'content_column' and 'metadata_column'

    This transform supports the following image formats: PDF, PNG, JPEG, TIFF, JPEG 2000, GIF, WebP, BMP, and PNM
    """

    COLUMN_NAMES = ["content_column"]

    def __init__(
        self,
        cache: BaseCache,
        output_columns: Dict[str, Any],
        url_column: str,
        lang: str = None,
    ) -> None:
        super().__init__(cache, output_columns)
        self.url_column = url_column
        self.lang = lang

        try:
            from PIL import Image
            import pytesseract
            from pdf2image import convert_from_path

            self.Image = Image
            self.pytesseract = pytesseract
            self.convert_from_path = convert_from_path
            self.pytesseract.get_tesseract_version()
        except ImportError:
            raise ImportError(
                "pillow, pytesseract, and pdf2image are required to use the ocr transform. Please install pillow, pytesseract, and pdf2image with the following command: pip install pillow pytesseract pdf2image"
            )
        except EnvironmentError:
            raise EnvironmentError(
                "The tesseract engine is required to use the ocr transform. Please see https://tesseract-ocr.github.io/tessdoc/Installation.html for installation instructions."
            )

    @staticmethod
    def name() -> str:
        return TransformType.OCR

    def get_image_ocr(self, image_or_image_path, lang: str = None) -> str:
        return self.pytesseract.image_to_string(image_or_image_path, lang=self.lang)

    def download_file(self, file_location: str) -> str:
        import os
        import tempfile
        import requests

        _, ext = os.path.splitext(file_location)
        temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)

        # Download file
        response = requests.get(file_location)
        response.raise_for_status()

        # Write to temp file
        with open(temp_file.name, "wb") as f:
            f.write(response.content)

        return temp_file.name

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """This function transforms an image into text using OCR.

        Args:
            row (Dict[str, Any]): The row of data to be transformed.

        Returns:
            Dict[str, Any]: The dict of output columns.
        """
        curr_file_location = row[self.url_column]
        # download file to temp location if a url
        try:
            curr_file_path = self.download_file(curr_file_location)
        except Exception as e:
            raise TransformError(
                TransformErrorType.TRANSFORM_ERROR,
                f"Error downloading file: {e}",
            )
        ocr_output = []
        if curr_file_path.endswith(".pdf"):
            pages = self.convert_from_path(curr_file_path)
            ocr_output = [self.get_image_ocr(page, lang=self.lang) for page in pages]
        else:
            ocr_output = [self.get_image_ocr(curr_file_path, lang=self.lang)]

        transformed_row = {
            self.output_columns["content_column"]: "\n\n".join(ocr_output),
        }
        return self._return_output_row(transformed_row)

    def params(self) -> Dict[str, Any]:
        return {
            "output_columns": self.output_columns,
            "url_column": self.url_column,
            "lang": self.lang,
        }

    def input_columns(self) -> List[str]:
        return [self.url_column]
