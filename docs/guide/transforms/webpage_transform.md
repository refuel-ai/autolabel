The Webpage transform supports loading and processing webpage urls. Given a url, this transform will send the request to load the webpage and then parse the webpage returned to collect the text to send to the LLM.

Use this transform yourself here in a Colab - [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PwrdBUUX1u4X2SWjgKYNxB11Gb7XEIZs#scrollTo=1f17f05a)

In order to use this transform, use the following steps:

## Installation

Use the following command to download all dependencies for the webpage transform. `beautifulsoup4` must be version `4.12.2` or higher.

```bash
pip install beautifulsoup4 httpx fake_useragent
```

Make sure to do this before running the transform.

## Parameters for this transform

1. `url_column: str (Required)`: The column to retrieve the url from. This is the webpage that will be loaded by the transform.
2. `timeout: int (Optional: Default = 5)`: The timeout to wait until for loading the webpage. The request to the webpage will timeout after this. We will log an error and send an empty response after the timeout is reached.
3. `headers: Dict[str,str] (Optional: Default = {})`: Any headers that need to be passed into the webpage load request. Underneath we use requests to get the webpage and the headers are passed to request.

## Using the transform

Below is an example of a webpage transform to extract text from a webpage:

```json
{
  ..., # other config parameters
  "transforms": [
    ..., # other transforms
    {
      "name": "webpage_transform",
      "params": {
        "url_column": "url"
      },
      "output_columns": {
        "content_column": "webpage_content",
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

This runs the transformation. We will see the content in the webpage_content column. Access this using `ds.df` in the AutolabelDataset.
