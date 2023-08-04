from typing import List, Dict

from autolabel.transforms import BaseTransform


class PDFOCRTransform(BaseTransform):
    def __init__(
        self,
        output_columns: List[str],
        file_path_column: str,
        page_format: str = "Page {page_num}: {page_content}",
        page_sep: str = "\n\n",
    ) -> None:
        """The output columns for this class should be in the order: [content_column, num_pages_column]"""
        super().__init__(output_columns)
        self.file_path_column = file_path_column
        self.page_format = page_format
        self.page_sep = page_sep

        try:
            from pdf2image import convert_from_path
            import pytesseract

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

    @staticmethod
    def name() -> str:
        return "pdf_ocr"

    async def _apply(self, row: Dict[str, any]) -> Dict[str, any]:
        """This function transforms a PDF file into a string of text using OCR.
        It uses the pdf2image library to convert the PDF into images and then uses
        pytesseract to convert the images into text. The text is then formatted
        according to the page_format and page_sep parameters and returned as a string.

        Args:
            row (Dict[str, any]): The row of data to be transformed.

        Returns:
            Dict[str, any]: The dict of output columns.
        """
        pages = self.convert_from_path(row[self.file_path_column])
        texts = []
        for idx, page in enumerate(pages):
            text = self.pytesseract.image_to_string(page)
            texts.append(self.page_format.format(page_num=idx + 1, page_content=text))
        output = self.page_sep.join(texts)
        return {
            self.output_columns[0]: output,
            self.output_columns[1]: len(texts),
        }
