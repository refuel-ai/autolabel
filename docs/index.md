# Welcome to Refuel Oracle

For full documentation visit [refuel.ai](https://www.refuel.ai).

## Installation

    git clone git@github.com:refuel-ai/refuel-oracle.git
    cd refuel-oracle
    pip install -r requirements.txt

## Environment Setup
Using refuel-oracle will usually require integrations with one or more model providers.
For this example, we will be using OpenAI’s API, so we will first need to install their SDK.
pip install openai


We will need to set the environmental variable in the terminal

    export OPENAI_API_KEY="..."


Alternatively, you could do this from within a Jupyter notebook or python instance:

    import os
    os.environ["OPENAI_API_KEY"] = "..."


If using Anthropic Models, we need to install the anthropic library

    pip install anthropic


And export the API Key in the environment variables

    export ANTHROPIC_API_KEY="..."


## Using Oracle for classifying unlabeled data with ChatGPT

The Oracle class is initialized with a configuration file, which defines the task you would like the LLM to perform on your data (i.e. classification, entity recognition, etc.)

    annotator = Oracle('examples/config_chatgpt.json')


Many sample configuration files can be found in the examples directory of the repository. These can act as a good starting template for other projects.

In this example, we are using ChatGPT to classify news articles into the appropriate category.

    annotator.annotate(
    dataset='examples/ag_news_filtered_labels_sampled.csv',
    max_items=100,
    )



## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
