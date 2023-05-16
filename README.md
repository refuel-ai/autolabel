# 🏷 Autolabel

_Notes: (1) Autolabel is in under active development. Expect some sharp edges and bugs. (2) This README and the docs website are under construction._

<div align="center" style="width:800px">

[![lint](https://github.com/refuel-ai/refuel-oracle/actions/workflows/black.yaml/badge.svg)](https://github.com/refuel-ai/refuel-oracle/actions/workflows/black.yaml/badge.svg) &nbsp;&nbsp; [![docs](https://github.com/refuel-ai/refuel-oracle/actions/workflows/docs.yaml/badge.svg)](https://docs.refuel.ai/) &nbsp;&nbsp; [![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/BDwamgzFxm) &nbsp;&nbsp; [![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/RefuelAI) &nbsp;&nbsp; [![License: MIT](https://badgen.net/badge/license/MIT/blue)](https://opensource.org/licenses/MIT)
</div>


## Quick Install

Once the package is published to PyPI:

`pip install refuel-autolabel`

Before the package is published to PyPI:
1. git clone https://github.com/refuel-ai/refuel-oracle.git
2. cd refuel-oracle
3. `pip install .`

To download benchmark datasets:

1. Either you can run the `get_data.py` script in `data/` directory: 
`python get_data.py` 

2. Or you can download a single zip file with all CSV files from here: https://drive.google.com/file/d/157z6pz7IgOsk9x9ObwcvpZgeBOapEV9C/view?usp=sharing and unzip it inside the `data/` directory 

## What is Autolabel?

Large language models have been trained on internet-scale data and are extremely good at content understanding, especially in a few-shot capacity (also called in-context learning). This means that an LLM can perform many tasks with just a few examples provided. This can be especially useful for auto-labeling data for many diverse tasks with just a few examples.

Autolabel is a Python package that lets users leverage Large Language Models (LLMs) for creating large, clean and diverse labeled datasets, a critical first step to building new AI applications.

## 🚀 Getting started

Conceptually, a labeling task has three components:
1. Labeling task guidelines
2. LLM that we will use for labeling
3. Dataset that we want to get labeled

Each of these components is specified in the library with a config. Example configs for each of these components are shared in `examples/configs` directory. 

Let's imagine we are building an ML model to flag toxic comments on an online social media platform. We have a dataset of comments that we'd like to get labeled first in order to train our downstream model. For this case, here's what the example dataset and configs will look like:
1. Labeling task config: `examples/configs/task_configs/civil_comments_classification.json`
2. LLM config: `examples/configs/llm_configs/anthropic.json`
3. Dataset config: `examples/configs/dataset_configs/civil_comments.json` (corresponding dataset CSV file is in `data/civil_comments_test.csv`)

First let's initialize the labeling agent and pass it the task and llm config:

```python

from autolabel import LabelingAgent

agent = LabelingAgent(
    'examples/configs/task_configs/civil_comments_classification.json',
    'examples/configs/llm_configs/anthropic.json'
)
```

Now, let's pass the dataset that we'd like to label, and see an example prompt that will be sent to the LLM: 
```python

agent.plan(
    'data/civil_comments_test.csv',
    'examples/configs/dataset_configs/civil_comments.json'
)
```

This prints:

```
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 175.31it/s]
Total Estimated Cost: $11.866
Number of examples to label: 2000
Average cost per example: $0.00593


A prompt example:

You are an expert at identifying toxic comments and understanding if a comment is sexually explicit, obscene, toxic, insults a person, demographic or race.
Your job is to correctly label the provided input example into one of the following 2 categories.
Categories:
toxic
not toxic


You will return the answer in JSON format with one key: {"label": "the correct label"}

Some examples with their output answers are provided below:
Example: It's ridiculous that these guys are being called 'protesters'. Being armed is a threat of violence, which makes them terrorists.
Output:
{"label": "toxic"}

Example: This is so cool. It's like, 'would you want your mother to read this??' Really great idea, well done!
Output:
{"label": "not toxic"}

Now I want you to label the following example: 
Example: Integrity means that you pay your debts. Does this apply to President Trump too?
Output:
```

Finally, we can run the labeling on a subset or entirety of the dataset:

```python
labels, output_df, metrics = agent.run(
    'data/civil_comments_test.csv',
    'examples/configs/dataset_configs/civil_comments.json',
    max_items=100
)
```

In addition to the dataframe, this will also output a file with the labels per row in `data/civil_comments_test_labeled.csv`

## End-to-end Examples

See:
1. `examples/example_banking.ipynb` 
2. `examples/example_run.ipynb` 

## 🛠️ Contributing

As an open-source project, we are extremely open to contributions, whether in the form of a new feature, bug fixes, better documentation or flagging issues.
