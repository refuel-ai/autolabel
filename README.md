<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/Autolabel_wt.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/Autolabel_blk.png">
  <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="docs/assets/Autolabel_blk.png">
</picture>

</picture>

<p align="center">
    <b>Clean, labeled data at the speed of thought</b>.
</p>

<div align="center" style="width:800px">

[![lint](https://github.com/refuel-ai/autolabel/actions/workflows/black.yaml/badge.svg)](https://github.com/refuel-ai/autolabel/actions/workflows/black.yaml/badge.svg) [![docs](https://github.com/refuel-ai/autolabel/actions/workflows/docs.yaml/badge.svg)](https://docs.refuel.ai/) ![Tests](https://github.com/refuel-ai/autolabel/actions/workflows/test.yaml/badge.svg) [![Discord](https://img.shields.io/discord/1098746693152931901)](https://discord.gg/fweVnRx6CU) [![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/RefuelAI) [![License: MIT](https://badgen.net/badge/license/MIT/blue)](https://opensource.org/licenses/MIT)
</div>

## Quick Install

`pip install refuel-autolabel`

## 🏷 What is Autolabel

Access to [large, clean and diverse](https://twitter.com/karpathy/status/1528443124577513472?lang=en) labeled datasets is a critical component for any machine learning effort to be successful. But data labeling is a manual and time-consuming process. State-of-the-art LLMs like GPT-4 are able to [automatically label data](https://arxiv.org/abs/2303.15056) with [high accuracy](https://arxiv.org/abs/2303.16854), and at a fraction of the cost and time.

Autolabel is a Python library to label, clean and enrich text datasets with any Large Language Models (LLM) of your choice. A few key features:

1. Label data for [NLP tasks](https://docs.refuel.ai/guide/tasks/classification_task/) such as classification, question-answering and named entity-recognition, entity matching and more.
2. Use commercial or open source [LLMs](https://docs.refuel.ai/guide/llms/llms/) from providers such as OpenAI, Anthropic, HuggingFace, Google and more.
3. Support for research-proven LLM techniques to boost label quality, such as few-shot learning and chain-of-thought prompting.
4. [Confidence estimation](https://docs.refuel.ai/guide/accuracy/confidence/) and explanations out of the box for every single output label
5. [Caching and state management](https://docs.refuel.ai/guide/reliability/state-management/) to minimize costs and experimentation time

## 🚀 Getting started

Autolabel provides a simple 3-step process for labeling data:

1. Specify the labeling guidelines and LLM model to use in a JSON config.
2. Dry-run to make sure the final prompt looks good.
3. Kick off a labeling run for your dataset!

Let's imagine we are building an ML model to analyze sentiment analysis of movie review. We have a dataset of moview reviews that we'd like to get labeled first. For this case, here's what the example dataset and configs will look like:

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
        "task_guidelines": "You are an expert at analyzing the sentiment of moview reviews. Your job is to classify the provided movie review into one of the following labels: {labels}",
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
agent.plan('examples/movie_reviews/dataset.csv')
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
You are an expert at analyzing the sentiment of moview reviews. Your job is to classify the provided movie review into one of the following labels: [positive, negative, neutral]

You will return the answer with just one element: "the correct label"

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
labels, output_df, metrics = agent.run('examples/movie_reviews/dataset.csv')
```

## 🙌 Contributing

Autolabel is a rapidly developing project. We welcome contributions in all forms - bug reports, pull requests and ideas for improving the library.

1. Join the conversation on [Discord](https://discord.gg/fweVnRx6CU)
2. Review the 🛣️ [Roadmap]() and contribute your ideas.
3. Grab an [open issue](https://github.com/refuel-ai/autolabel/issues) on Github, and submit a [pull request](https://github.com/refuel-ai/autolabel/blob/main/CONTRIBUTING.md).
