<img alt="Refuel logo" src="https://raw.githubusercontent.com/refuel-ai/autolabel/main/docs/assets/Autolabel_blk_w_background.png">

<h4 align="center">
  <a href="https://discord.gg/fweVnRx6CU">Discord</a> |
  <a href="https://twitter.com/RefuelAI">Twitter</a> |
  <a href="https://www.refuel.ai/">Website</a> |
  <a href="https://www.refuel.ai/blog-posts/llm-labeling-technical-report">Benchmark</a>
</h4>

<div align="center" style="width:800px">

[![lint](https://github.com/refuel-ai/autolabel/actions/workflows/black.yaml/badge.svg)](https://github.com/refuel-ai/autolabel/actions/workflows/black.yaml/badge.svg) ![Tests](https://github.com/refuel-ai/autolabel/actions/workflows/test.yaml/badge.svg) ![Commit Activity](https://img.shields.io/github/commit-activity/m/refuel-ai/autolabel) [![Discord](https://img.shields.io/discord/1098746693152931901)](https://discord.gg/fweVnRx6CU) [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1t-9vNLkyoyySAG_0w3eR98biBOXlMO-E?usp=sharing)

</div>

## ⚡ Quick Install

`pip install refuel-autolabel`

## 📖 Documentation

[https://docs.refuel.ai/](https://docs.refuel.ai/)

## 🏷 What is Autolabel

Access to [large, clean and diverse](https://twitter.com/karpathy/status/1528443124577513472?lang=en) labeled datasets is a critical component for any machine learning effort to be successful. State-of-the-art LLMs like GPT-4 are able to [automatically label data](https://arxiv.org/abs/2303.15056) with [high accuracy](https://arxiv.org/abs/2303.16854), and at a fraction of the cost and time compared to manual labeling.

Autolabel is a Python library to label, clean and enrich text datasets with any Large Language Models (LLM) of your choice.

## 🌟 (New!) Benchmark models on Refuel's Benchmark

You can

## 🌟 Access RefuelLLM through Autolabel

You can access RefuelLLM, our recently announced LLM purpose built for data labeling, through Autolabel (Read more about it in this [blog post](http://www.refuel.ai/blog-posts/announcing-refuel-llm)). RefuelLLM is a Llama-v2-13b base model, instruction tuned on over 2500 unique (5.24B tokens) labeling tasks spanning categories such as classification, entity resolution, matching, reading comprehension and information extraction. You can experiment with the model in the playground [here](https://app.refuel.ai/playground).

<img alt="Refuel Performance" src="https://raw.githubusercontent.com/refuel-ai/autolabel/main/docs/assets/refuel_llm_performance.png">

You can request access to RefuelLLM [here](https://refuel-ai.typeform.com/llm-access). Read the docs about using RefuelLLM in autolabel [here](https://docs.refuel.ai/guide/llms/llms/#refuel).

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
            "neutral"
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

from autolabel import LabelingAgent, AutolabelDataset

agent = LabelingAgent(config='config.json')
```

Preview an example prompt that will be sent to the LLM:

```python
ds = AutolabelDataset('dataset.csv', config = config)
agent.plan(ds)
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
ds = agent.run(ds)
```

The output dataframe contains the label column:

```python
ds.df.head()
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

Check out our [technical report]() to learn more about the performance of RefuelLLM-v2 on our benchmark. You can replicate the benchmark yourself by following the steps below

```python
cd autolabel/benchmark
python benchmark.py --model $model --base_dir benchmark-results
python results.py --eval_dir benchmark-results
cat results.csv
```

You can benchmark the relevant model by replacing $model with the name of the model needed to be benchmarked. If it is an API hosted model like `gpt-3.5-turbo`, `gpt-4-1106-preview`, `claude-3-opus-20240229`, `gemini-1.5-pro-preview-0409` or some other Autolabel supported model, just write the name of the model. If the model to be benchmarked is a [vLLM supported model](https://docs.vllm.ai/en/latest/models/supported_models.html) then pass the local path or the huggingface path corresponding to the model. This will run the benchmark along with the _same_ prompts for all models.

The `results.csv` will contain a row with every model that was benchmarked as a row. Look at `benchmark/results.csv` for an example.

## 🛠️ Roadmap

Check out our [public roadmap](https://github.com/orgs/refuel-ai/projects/15) to learn more about ongoing and planned improvements to the Autolabel library.

We are always looking for suggestions and contributions from the community. Join the discussion on [Discord](https://discord.gg/fweVnRx6CU) or open a [Github issue](https://github.com/refuel-ai/autolabel/issues) to report bugs and request features.

## 🙌 Contributing

Autolabel is a rapidly developing project. We welcome contributions in all forms - bug reports, pull requests and ideas for improving the library.

1. Join the conversation on [Discord](https://discord.gg/fweVnRx6CU)
2. Open an [issue](https://github.com/refuel-ai/autolabel/issues) on Github for bugs and request features.
3. Grab an open issue, and submit a [pull request](https://github.com/refuel-ai/autolabel/blob/main/CONTRIBUTING.md).
