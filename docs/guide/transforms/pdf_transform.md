The PDF transform allows users to extract text from pdf files. Autolabel offers both direct text extraction, useful for extracting text from pdfs that contain text, and optical character recognition (OCR) text extraction, useful for extracting text from pdfs that contain images. To use this transform, follow these steps:

<ol>
<li>Install dependencies
   For direct text extraction, install the <code>pdfplumber</code> package:

```bash
pip install pdfplumber
```

For OCR text extraction, install the <code>pdf2image</code> and <code>pytesseract</code> packages:

```bash
pip install pdf2image pytesseract
```

</li>
<li>Add the transform to your config file

below is an example of a pdf transform to extract text from a pdf file:

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

The `params` field contains the following parameters:

<ul>
<li>file_path_column: the name of the column containing the file paths of the pdf files to extract text from</li>
<li>ocr_enabled: a boolean indicating whether to use OCR text extraction or not</li>
<li>page_format: a string containing the format to use for each page of the pdf file. The following fields can be used in the format string:
<ul>
<li>page_num: the page number of the page</li>
<li>page_content: the content of the page</li></li>
</ul>
<li>page_sep: a string containing the separator to use between each page of the pdf file
</ul>

For example, if the pdf file contained 2 pages with "Hello," on the first page and "World!" on the second, a page_format of <code>{page_num} - {page_content}</code> and a page_sep of <code>\n</code> would result in the following output:

```python
"1 - Hello,\n2 - World!"
```

The metadata column contains a dict with the field "num_pages" indicating the number of pages in the pdf file.

</li>
<li>Run the transform

```python
from autolabel import LabelingAgent, AutolabelDataset
agent = LabelingAgent(config)
ds = agent.transform(ds)
```

This runs the transformation. We will see the content in the correct column. Access this using `ds.df` in the AutolabelDataset.
