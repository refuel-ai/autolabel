from typing import Any, Dict, List

from autolabel.cache import BaseCache
from autolabel.transforms import BaseTransform
from autolabel.transforms.schema import TransformType


class PDFTransform(BaseTransform):
    """This class is used to extract text from PDFs. The output columns dictionary for this class should include the keys 'content_column' and 'metadata_column'"""

    COLUMN_NAMES = [
        "content_column",
        "metadata_column",
    ]

    def __init__(
        self,
        cache: BaseCache,
        output_columns: Dict[str, Any],
        file_path_column: str,
        ocr_enabled: bool = False,
        page_format: str = "Page {page_num}: {page_content}",
        page_sep: str = "\n\n",
        lang: str = None,
    ) -> None:
        super().__init__(cache, output_columns)
        self.file_path_column = file_path_column
        self.ocr_enabled = ocr_enabled
        self.page_format = page_format
        self.page_sep = page_sep
        self.lang = lang

        if self.ocr_enabled:
            try:
                import pytesseract
                from pdf2image import convert_from_path

                self.convert_from_path = convert_from_path
                self.pytesseract = pytesseract
                self.pytesseract.get_tesseract_version()
            except ImportError:
                raise ImportError(
                    "pdf2image and pytesseract are required to use the pdf transform with ocr. Please install pdf2image and pytesseract with the following command: pip install pdf2image pytesseract"
                )
            except EnvironmentError:
                raise EnvironmentError(
                    "The tesseract engine is required to use the pdf transform with ocr. Please see https://tesseract-ocr.github.io/tessdoc/Installation.html for installation instructions."
                )
        else:
            try:
                from langchain_community.document_loaders import PDFPlumberLoader

                self.PDFPlumberLoader = PDFPlumberLoader
            except ImportError:
                raise ImportError(
                    "pdfplumber is required to use the pdf transform. Please install pdfplumber with the following command: pip install pdfplumber"
                )

    @staticmethod
    def name() -> str:
        return TransformType.PDF

    def get_page_texts(self, row: Dict[str, Any]) -> List[str]:
        """This function gets the text from each page of a PDF file.
        If OCR is enabled, it uses the pdf2image library to convert the PDF into images and then uses
        pytesseract to convert the images into text. Otherwise, it uses pdfplumber to extract the text.

        Args:
            row (Dict[str, Any]): The row of data to be transformed.

        Returns:
            List[str]: A list of strings containing the text from each page of the PDF.
        """
        if self.ocr_enabled:
            pages = self.convert_from_path(row[self.file_path_column])
            return [
                self.pytesseract.image_to_string(page, lang=self.lang) for page in pages
            ]
        else:
            loader = self.PDFPlumberLoader(row[self.file_path_column])
            return [page.page_content for page in loader.load()]

    async def _apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """This function transforms a PDF file into a string of text.
        The text is formatted according to the page_format and
        page_sep parameters and returned as a string.

        Args:
            row (Dict[str, Any]): The row of data to be transformed.

        Returns:
            Dict[str, Any]: The dict of output columns.
        """
        texts = []
        for idx, text in enumerate(self.get_page_texts(row)):
            texts.append(self.page_format.format(page_num=idx + 1, page_content=text))
        output = self.page_sep.join(texts)
        transformed_row = {
            self.output_columns["content_column"]: output,
            self.output_columns["metadata_column"]: {"num_pages": len(texts)},
        }
        return self._return_output_row(transformed_row)

    def params(self):
        return {
            "file_path_column": self.file_path_column,
            "ocr_enabled": self.ocr_enabled,
            "page_header": self.page_format,
            "page_sep": self.page_sep,
            "output_columns": self.output_columns,
            "lang": self.lang,
        }

    def input_columns(self) -> List[str]:
        return [self.file_path_column]