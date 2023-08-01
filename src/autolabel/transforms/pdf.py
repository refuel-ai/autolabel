from typing import List, Dict, Any

from langchain.document_loaders import PyPDFLoader

from autolabel.transforms import BaseTransform


class PDFTransform(BaseTransform):
    def __init__(
        self,
        output_columns: List[str],
        file_path_column: str,
        page_header: str = "Page {page_num}: ",
        page_sep: str = "\n\n",
    ) -> None:
        """The output columns for this class should be in the order: [content_column, num_pages_column]"""
        super().__init__(output_columns)
        self.file_path_column = file_path_column
        self.page_header = page_header
        self.page_sep = page_sep

    @staticmethod
    def name() -> str:
        return "pdf"

    def transform(self, row: Dict[str, any]) -> Dict[str, any]:
        """This function transforms a PDF file into a string of text. It uses the PyPDFLoader to load and split the PDF into pages.
        Each page is then converted into text and appended to the output string.

        Args:
            row (Dict[str, any]): The row of data to be transformed.

        Returns:
            Dict[str, any]: The transformed row of data.
        """
        loader = PyPDFLoader(row[self.file_path_column])
        page_contents = []
        for idx, page in enumerate(loader.load_and_split()):
            page_contents.append(
                self.page_header.format(page_num=idx + 1) + page.page_content
            )
        output = self.page_sep.join(page_contents)
        return {
            self.output_columns[0]: output,
            self.output_columns[1]: len(page_contents),
        }
