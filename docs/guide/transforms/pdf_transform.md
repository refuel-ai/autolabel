The PDF transform allows users to extract text from pdf files. Autolabel offers both direct text extraction, useful for extracting text from pdfs that contain text, and optical character recognition (OCR) text extraction, useful for extracting text from pdfs that contain images. To use this transform, follow these steps:

## Installation

For direct text extraction, install the <code>pdfplumber</code> package:

```bash
pip install pdfplumber
```

For OCR text extraction, install the <code>pdf2image</code> and <code>pytesseract</code> packages:

```bash
pip install pdf2image pytesseract
```

The tesseract engine is also required for OCR text extraction. See the [tesseract docs](https://tesseract-ocr.github.io/tessdoc/Installation.html) for installation instructions.

## Parameters for this transform

<ol>
<li>file_path_column: the name of the column containing the file paths of the pdf files to extract text from</li>
<li>ocr_enabled: a boolean indicating whether to use OCR text extraction or not</li>
<li>page_format: a string containing the format to use for each page of the pdf file. The following fields can be used in the format string:
<ul>
<li>page_num: the page number of the page</li>
<li>page_content: the content of the page</li>
</ul></li>
<li>page_sep: a string containing the separator to use between each page of the pdf file</li>
<li>lang: a string indicating the language of the text in the pdf file. See the [tesseract docs](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html) for a full list of supported languages</li>
</ol>

### Output Format

The page_format and page_sep parameters define how the text extracted from the pdf will be formatted. For example, if the pdf file contained 2 pages with "Hello," on the first page and "World!" on the second, a page_format of <code>{page_num} - {page_content}</code> and a page_sep of <code>\n</code> would result in the following output:

```python
"1 - Hello,\n2 - World!"
```

The metadata column contains a dict with the field "num_pages" indicating the number of pages in the pdf file.

## Using the transform

Below is an example of a pdf transform to extract text from a pdf file:

```json
{
  ..., # other config parameters
  "transforms": [
    ..., # other transforms
    {
      "name": "pdf",
      "params": {
        "file_path_column": "file_path",
        "ocr_enabled": true,
        "page_format": "Page {page_num}: {page_content}",
        "page_sep": "\n\n"
      },
      "output_columns": {
        "content_column": "content",
        "metadata_column": "metadata"
      }
    }
  ]
}
```

## Run the transform

```python
from autolabel import LabelingAgent, AutolabelDataset
agent = LabelingAgent(config)
ds = agent.transform(ds)
```

This runs the transformation. We will see the content in the correct column. Access this using `ds.df` in the AutolabelDataset.
