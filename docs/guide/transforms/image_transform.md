The image transform allows users to extract text from image files. Autolabel uses optical character recognition (OCR) to read the images. To use this transform, follow these steps:

## Installation

Use the following command to download all dependencies for the image transform.

```bash
pip install pillow pytesseract
```

The tesseract engine is also required for OCR text extraction. See the [tesseract docs](https://tesseract-ocr.github.io/tessdoc/Installation.html) for installation instructions.

## Parameters for this transform

1. file_path_column: the name of the column containing the file paths of the pdf files to extract text from
2. lang: a string indicating the language of the text in the pdf file. See the [tesseract docs](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html) for a full list of supported languages

## Using the transform

Below is an example of an image transform to extract text from an image file:

```json
{
  ..., # other config parameters
  "transforms": [
    ..., # other transforms
    {
      "name": "image",
      "params": {
        "file_path_column": "file_path",
        "lang": "eng"
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
