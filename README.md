<img alt="Refuel logo" src="https://raw.githubusercontent.com/refuel-ai/autolabel/main/docs/assets/Autolabel_blk_w_background.png">


<h4 align="center">
  <a href="https://docs.refuel.ai/">Docs</a> |
  <a href="https://discord.gg/fweVnRx6CU">Discord</a> |
  <a href="https://twitter.com/RefuelAI">Twitter</a> |
  <a href="https://www.refuel.ai/">Website</a>
</h4>

<div align="center" style="width:800px">

[![lint](https://github.com/refuel-ai/autolabel/actions/workflows/black.yaml/badge.svg)](https://github.com/refuel-ai/autolabel/actions/workflows/black.yaml/badge.svg) ![Tests](https://github.com/refuel-ai/autolabel/actions/workflows/test.yaml/badge.svg) ![Commit Activity](https://img.shields.io/github/commit-activity/m/refuel-ai/autolabel) [![Discord](https://img.shields.io/discord/1098746693152931901)](https://discord.gg/fweVnRx6CU) [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1t-9vNLkyoyySAG_0w3eR98biBOXlMO-E?usp=sharing)
</div>

## ⚡ Quick Install

`pip install refuel-autolabel`

## 🏷 What is Autolabel

Access to [large, clean and diverse](https://twitter.com/karpathy/status/1528443124577513472?lang=en) labeled datasets is a critical component for any machine learning effort to be successful. State-of-the-art LLMs like GPT-4 are able to [automatically label data](https://arxiv.org/abs/2303.15056) with [high accuracy](https://arxiv.org/abs/2303.16854), and at a fraction of the cost and time compared to manual labeling.

Autolabel is a Python library to label, clean and enrich text datasets with any Large Language Models (LLM) of your choice.

## 🚀 Getting started

Autolabel provides a simple 3-step process for labeling data:

1. Specify the labeling guidelines and LLM model to use in a JSON config.
2. Dry-run to make sure the final prompt looks good.
3. Kick off a labeling run for your dataset!

Let's imagine we are building an ML model to analyze sentiment analysis of movie review. We have a dataset of movie reviews that we'd like to get labeled first. For this case, here's what the example dataset and configs will look like:

```python
{
    "task_name": "MovieSentimentReview",
    "task_type": "classification",
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "prompt": {
        "task_guidelines": "You are an expert at analyzing the sentiment of movie reviews. Your job is to classify the provided movie review into one of the following labels: {labels}",
        "labels": [
            "positive",
            "negative",
            "neutral",
        ],
        "few_shot_examples": [
            {
                "example": "I got a fairly uninspired stupid film about how human industry is bad for nature.",
                "label": "negative"
            },
            {
                "example": "I loved this movie. I found it very heart warming to see Adam West, Burt Ward, Frank Gorshin, and Julie Newmar together again.",
                "label": "positive"
            },
            {
                "example": "This movie will be played next week at the Chinese theater.",
                "label": "neutral"
            }
        ],
        "example_template": "Input: {example}\nOutput: {label}"
    }
}
```

Initialize the labeling agent and pass it the config:

```python

from autolabel import LabelingAgent

agent = LabelingAgent(config='config.json')
```

Preview an example prompt that will be sent to the LLM:

```python
agent.plan('dataset.csv')
```

This prints:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100/100 0:00:00 0:00:00
┌──────────────────────────┬─────────┐
│ Total Estimated Cost     │ $0.538  │
│ Number of Examples       │ 200     │
│ Average cost per example │ 0.00269 │
└──────────────────────────┴─────────┘
─────────────────────────────────────────

Prompt Example:
You are an expert at analyzing the sentiment of movie reviews. Your job is to classify the provided movie review into one of the following labels: [positive, negative, neutral]

Some examples with their output answers are provided below:

Example: I got a fairly uninspired stupid film about how human industry is bad for nature.
Output:
negative

Example: I loved this movie. I found it very heart warming to see Adam West, Burt Ward, Frank Gorshin, and Julie Newmar together again.
Output:
positive

Example: This movie will be played next week at the Chinese theater.
Output:
neutral

Now I want you to label the following example:
Input: A rare exception to the rule that great literature makes disappointing films.
Output:

─────────────────────────────────────────────────────────────────────────────────────────

```

Finally, we can run the labeling on a subset or entirety of the dataset:

```python
labels, output_df, metrics = agent.run('dataset.csv')
```

The output dataframe contains the label column:

```python
output_df.head()
                                                text  ... MovieSentimentReview_llm_label
0  I was very excited about seeing this film, ant...  ...                       negative
1  Serum is about a crazy doctor that finds a ser...  ...                       negative
4  I loved this movie. I knew it would be chocked...  ...                       positive
...
```

## Features

1. Label data for [NLP tasks](https://docs.refuel.ai/guide/tasks/classification_task/) such as classification, question-answering and named entity-recognition, entity matching and more.
2. Use commercial or open source [LLMs](https://docs.refuel.ai/guide/llms/llms/) from providers such as OpenAI, Anthropic, HuggingFace, Google and more.
3. Support for research-proven LLM techniques to boost label quality, such as few-shot learning and chain-of-thought prompting.
4. [Confidence estimation](https://docs.refuel.ai/guide/accuracy/confidence/) and explanations out of the box for every single output label
5. [Caching and state management](https://docs.refuel.ai/guide/reliability/state-management/) to minimize costs and experimentation time

## Access to Refuel hosted LLMs

Refuel provides access to hosted open source LLMs for labeling, and for estimating confidence This is helpful, because you can calibrate a confidence threshold for your labeling task, and then route less confident labels to humans, while you still get the benefits of auto-labeling for the confident examples.

In order to use Refuel hosted LLMs, you can [request access here](https://refuel-ai.typeform.com/llm-access).

## Benchmark

Check out our [technical report](https://www.refuel.ai/blog-posts/llm-labeling-technical-report) to learn more about the performance of various LLMs, and human annoators, on label quality, turnaround time and cost.

## 🛠️ Roadmap
Our goal is to allow users to label, create or enrich any dataset, with any LLM - easily and quickly.

There are four focus areas for Autolabel for 2023:

* **Tasks**: Add support for tasks such as retrieval, attribute enrichment and text generation/writing.
* **LLMs**: Add support for more LLMs, especially open source models like Falcon, MPT and Dolly and the ability to plug in your own LLMs. 
* Workflows for **experimenting with your datasets** more easily: Add support for richer data types (such as PDFs and HTML documents) and the ability to run benchmarking on your data sources.
* Techniques to **improve labeling accuracy**: Add support for automatic prompt improvement and tools for better error analysis to iteratively improve LLM performance on different tasks 

We will be releasing a more detailed roadmap soon, but we love suggestions and contributions from the community. Chat with the Refuel team and Autolabel community on [Discord](https://discord.gg/fweVnRx6CU) or open [Github issues](https://github.com/refuel-ai/autolabel/issues) to report bugs and request features.

## 🙌 Contributing

Autolabel is a rapidly developing project. We welcome contributions in all forms - bug reports, pull requests and ideas for improving the library.

1. Join the conversation on [Discord](https://discord.gg/fweVnRx6CU)
2. Open an [issue](https://github.com/refuel-ai/autolabel/issues) on Github for bugs and request features.
3. Grab an open issue, and submit a [pull request](https://github.com/refuel-ai/autolabel/blob/main/CONTRIBUTING.md).
