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

A dataset[^1] containing 200 unlabeled movie reviews is available [here](https://github.com/refuel-ai/autolabel/blob/main/docs/assets/movie_reviews_preview.csv), and a couple of examples (with labels) are shown below:

{{ read_csv('docs/assets/movie_reviews_preview.csv') }}

Our goal is to label the full 200 examples using Autolabel. 

[^1]: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). [Learning Word Vectors for Sentiment Analysis](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf). The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

## Labeling with AutoLabel

Autolabel provides a simple 3-step process for labeling data:

* Specify the configuration of your labeling task as a JSON
* Preview the labeling task against your dataset
* Label your data!

### Specify the labeling task via configuration

First, create a JSON file that specifies:

* Task: `task_name` is `MovieSentimentReview` and the `task_type` is `classification`
* LLM: Choice of LLM provider and model - here we are using `gpt-3.5-turbo` from OpenAI
* Instructions: These are the labeling guidelines provided to the LLM for labeling

```python
config = {
    "task_name": "MovieSentimentReview",
    "task_type": "classification",
    "dataset": {
        "label_column": "label"
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
        ],
        "example_template": "Example: {text}\nLabel: {label}"
    }
}
```

### Preview the labeling against your dataset

First import `autolabel`, create a `LabelingAgent` object and then run the `plan` command against the dataset (available [here](https://docs.refuel.ai/guide/resources/refuel_datasets/) and can be downloaded through the `autolabel.get_data` function):

```python
from autolabel import LabelingAgent, get_data
get_data('movie_reviews')

agent = LabelingAgent(config)
agent.plan('test.csv')
```

This produces:
```
Computing embeddings... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100/100 0:00:00 0:00:00
┌──────────────────────────┬─────────┐
│ Total Estimated Cost     │ $0.538  │
│ Number of Examples       │ 200     │
│ Average cost per example │ 0.00269 │
└──────────────────────────┴─────────┘
───────────────────────────────────────────── Prompt Example ─────────────────────────────────────────────
You are an expert at analyzing the sentiment of moview reviews. Your job is to classify the provided movie review as positive or negative.

You will return the answer with just one element: "the correct label"

Now I want you to label the following example:
Input: I was very excited about seeing this film, anticipating a visual excursus on the relation of artistic beauty and nature, containing the kinds of wisdom the likes of "Rivers and Tides." However, that's not what I received. Instead, I get a fairly uninspired film about how human industry is bad for nature. Which is clearly a quite unorthodox claim.<br /><br />The photographer seems conflicted about the aesthetic qualities of his images and the supposed "ethical" duty he has to the workers occasionally peopling the images, along the periphery. And frankly, the images were not generally that impressive. And according to this "artist," scale is the basis for what makes something beautiful.<br /><br />In all respects, a stupid film. For people who'd like to feel better about their environmental consciousness ... but not for any one who would like to think about the complexities of the issues surrounding it.
Output:
──────────────────────────────────────────────────────────────────────────────────────────────────────────
```

This shows you:

* Number of examples to be labeled in the dataset: `200`
* Estimated cost of running this labeling task: `<$1`
* Exact prompt being sent to the LLM

Having previewed the labeling, we are ready to start labeling. 


### Label your dataset

Now, you can use the `run` command to label:

```python
labels, output_df, metrics = agent.run('docs/assets/movie_reviews.csv')

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200/200 0:04:01 0:00:00
```

This takes just a few minutes to run, and returns the labeled data as a Pandas DataFrame (`output_df` here). We can explore this by running:
```python
output_df.head()
>
                                                text  ... MovieSentimentReview_llm_label
0  I was very excited about seeing this film, ant...  ...                       negative
1  Serum is about a crazy doctor that finds a ser...  ...                       negative
2  This movie was so very badly written. The char...  ...                       negative
3  Hmmmm, want a little romance with your mystery...  ...                       negative
4  I loved this movie. I knew it would be chocked...  ...                       positive

[5 rows x 4 columns]
```

At this point, we have a labeled dataset ready, and we can begin training our ML models. 

## Summary

In this simple walkthrough, we have installed `autolabel`, gone over an example dataset to label (sentiment analysis for moview reviews) and used `autolabel` to label this dataset in just a few minutes. 

We hope that this gives you a glimpse of what you can do with Refuel. There are many other [labeling tasks](/guide/tasks/tasks) available within Autolabel, and if you have any questions, join our community <a href="https://discord.gg/uEdr8nrMGm" target="_blank">here</a> or [open an issue](https://github.com/refuel-ai/autolabel/issues/new/choose) on [Github](https://github.com/refuel-ai/autolabel). 
