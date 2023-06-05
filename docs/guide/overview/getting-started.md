# Getting Started with Autolabel
This page will walk you through your very first labeling task using Refuel Autolabel. Specifically, it'll go over:

* Installation
* Overview of a dataset to label
* Labeling the dataset using Autolabel

## Installation

Autolabel is available on PyPI and can be installed by running:
``` bash
pip install refuel-autolabel
```

Separate from the Autolabel library, you'll also need to install an integration with your favorite LLM provider. In the example below, we'll be using OpenAI, so you'll need to install the OpenAI SDK and set your API key as an environment variable:
```bash
pip install openai
export OPENAI_API_KEY="<your-openai-key>"
```

To use a different LLM provider, follow the documentation [here](/guide/llms/llms). 

## Goal: Sentiment Analysis on a Movie Review Dataset
Let's say we wanted to run sentiment analysis on a dataset of movie reviews. We want to train our own ML model, but first, we need to label some data for training.

Now, we could label a few hundred examples by hand which would take us a few hours. Instead, let's use Autolabel to get a clean, labeled dataset in a few minutes. 

A dataset [footnote needed] containing 200 unlabeled movie reviews is available here [link needed], and a couple of examples (with labels) are shown below:

{{ read_csv('docs/assets/movie_reviews_preview.csv') }}

Our goal is to label the full 200 examples using Autolabel. 

## Labeling with AutoLabel

### Specify the labeling task via configuration

First, create a JSON file that specifies:

* Task: `task_name` is `MovieSentimentReview` and the `task_type` is `classification`
* Instructions: These are the labeling guidelines provided to the LLM for labeling
* LLM: Choice of LLM provider and model - here we are using `gpt-3.5-turbo` from OpenAI

```python
config = {
    "task_name": "MovieSentimentReview",
    "task_type": "classification",
    "dataset": {
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "prompt": {
        "task_guidelines": "You are an expert at analyzing the sentiment of moview reviews. Your job is to classify the provided movie review as positive or negative.",
        "labels": [
            "positive",
            "negative"
        ]
    }
}
```

### Plan

### Label
