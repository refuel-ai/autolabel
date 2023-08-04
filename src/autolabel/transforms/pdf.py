from typing import List, Dict, Any

from autolabel.transforms import BaseTransform


class PDFTransform(BaseTransform):
    def __init__(
        self,
        output_columns: List[str],
        file_path_column: str,
        page_header: str = "Page {page_num}: {page_content}",
        page_sep: str = "\n\n",
    ) -> None:
        """The output columns for this class should be in the order: [content_column, num_pages_column]"""
        super().__init__(output_columns)
        self.file_path_column = file_path_column
        self.page_format = page_header
        self.page_sep = page_sep

        try:
            from langchain.document_loaders import PDFPlumberLoader

            self.PDFPlumberLoader = PDFPlumberLoader
        except ImportError:
            raise ImportError(
                "pdfplumber is required to use the pdf transform. Please install pdfplumber with the following command: pip install pdfplumber"
            )

    @staticmethod
    def name() -> str:
        return "pdf"

    async def _apply(self, row: Dict[str, any]) -> Dict[str, any]:
        """This function transforms a PDF file into a string of text.
        It uses the pdfplumber library to convert the PDF into text.
        The text is then formatted according to the page_format and
        page_sep parameters and returned as a string.

        Args:
            row (Dict[str, any]): The row of data to be transformed.

        Returns:
            Dict[str, any]: The dict of output columns.
        """
        loader = self.PDFPlumberLoader(row[self.file_path_column])
        texts = []
        for idx, page in enumerate(loader.load()):
            text = page.page_content
            texts.append(self.page_format.format(page_num=idx + 1, page_content=text))
        output = self.page_sep.join(texts)
        return {
            self.output_columns[0]: output,
            self.output_columns[1]: len(texts),
        }
