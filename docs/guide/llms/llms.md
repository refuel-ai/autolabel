# Large Language Models (LLMs)

Autolabel supports multiple LLMs for labeling data. Some LLMs are available by calling an API with the appropriate API keys (OpenAI, Anthropic, etc.) while others can be run locally (such as the ones available on Huggingface). The LLM used to label can be controlled using the `provider` and `name` keys in the dictionary specified under `model` in the input config. 

Each LLM belongs to an LLM provider -- which refers to the organization or open-source framework through which we are able to access the LLM. A full list of LLM providers and LLMs that are currently supported is provided towards the end of the document.

Autolabel makes it easy to try out different LLMs for your task and this page will walk you through how to get started with each LLM provider and model. Separately, we've also benchmarked multiple LLMs across different datasets - you can read the full technical report here [link to blog post] or check out the latest benchmark results here [link to benchmarks]. 

## OpenAI
To use models from [OpenAI](https://platform.openai.com/docs/models), you can set `provider` to `openai` when creating a labeling configuration. The specific model that will be queried can be specified using the `name` key. Autolabel currently supports the following models from OpenAI:

* `text-davinci-003`
* `gpt-3.5-turbo`
* `gpt-4`  (8K Context)

`gpt-4` (8K Context) is the most capable (and most expensive) model from OpenAI, while `gpt-3.5-turbo` is the cheapest (but still quite capable). `gpt-4` costs 15 times `gpt-3.5-turbo`, at $0.03/1K tokens while `gpt-3.5-turbo` costs only $0.002/1K tokens. `text-davinci-003` model on the other hand costs $0.02/1K tokens. Detailed pricing for these models is available [here](https://openai.com/pricing). 

### Setup
To use OpenAI models with Autolabel, make sure to first install the relevant packages by running:
```bash
pip install refuel-autolabel[openai]
```
and also setting the following environment variable:
```
export OPENAI_API_KEY=<your-openai-key>
```
replacing `<your-openai-key>` with your API key, which you can get from [here](https://platform.openai.com/account/api-keys).

### Example usage
Here is an example of setting config to a dictionary that will use openai's `gpt-3.5-turbo` model for labeling. Specifically, note that in the dictionary proivded by the `model` tag, `provider` is set to `openai` and `name` is set to be `gpt-3.5-turbo`. `name` can be switched to use any of the three models mentioned above.

```python
config = {
    "task_name": "OpenbookQAWikipedia",
    "task_type": "multi_choice_question_answering",
    "dataset": {
        "label_column": "answer",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo",
        "params": {}
    },
    "prompt": {
        "task_guidelines": "You are an expert at answering questions."
        "example_template": "Question: {question}\nAnswer: {answer}"
    }
}
```

### Additional parameters
A few parameters can be passed in alongside `openai` models to tweak their behavior:

* `max_tokens` (int): The maximum tokens to sample from the model
* `temperature` (float): A float between 0 and 2 which indicates the diversity you want in the output. 0 uses greedy sampling (picks the most likely outcome).

These parameters can be passed in via the `params` dictionary under `model`. Here is an example:
```python
"model": {
    "provider": "openai",
    "name": "gpt-3.5-turbo",
    "params": {
        "max_tokens": 512,
        "temperature": 0.1
    }
}
```

## Anthropic
To use models from [Anthropic](https://www.anthropic.com/index/introducing-claude), you can set the `provider` to `anthropic` when creating a labeling configuration. The specific model that will be queried can be specified using the `name` key. Autolabel currently supports the following models from Anthropic:

* `claude-instant-v1`
* `claude-v1`

`claude-v1` is a state-of-the-art high-performance model, while `claude-instant-v1` is a lighter, less expensive, and much faster option. `claude-instant-v1` is ~6.7 times cheaper than `claude-v1`, at $1.63/1 million tokens. On the other hand `claude-v1` costs $11.02/1 million tokens.

### Setup
To use Anthropic models with Autolabel, make sure to first install the relevant packages by running:
```bash
pip install refuel-autolabel[anthropic]
```
and also setting the following environment variable:
```
export ANTHROPIC_API_KEY=<your-anthropic-key>
```
replacing `<your-anthropic-key>` with your API key, which you can get from [here](https://console.anthropic.com/docs/access).

### Example usage
Here is an example of setting config to a dictionary that will use anthropic's `claude-instant-v1` model for labeling. Specifically, note that in the dictionary proivded by the `model` tag, `provider` is set to `anthropic` and `name` is set to be `claude-instant-v1`. `name` can be switched to use any of the two models mentioned above.

```python
config = {
    "task_name": "OpenbookQAWikipedia",
    "task_type": "multi_choice_question_answering",
    "dataset": {
        "label_column": "answer",
        "delimiter": ","
    },
    "model": {
        "provider": "anthropic",
        "name": "claude-instant-v1",
        "params": {}
    },
    "prompt": {
        "task_guidelines": "You are an expert at answering questions."
        "example_template": "Question: {question}\nAnswer: {answer}"
    }
}
```
### Additional parameters
A few parameters that can be passed in for `anthropic` models to control the model behavior:

* `max_tokens_to_sample` (int): The maximum tokens to sample from the model
* `temperature` (float): A float between 0 and 2 which indicates the diversity you want in the output. 0 uses greedy sampling (picks the most likely outcome).

These parameters can be passed in via the `params` dictionary under `model`. Here is an example:
```python
"model": {
    "provider": "anthropic",
    "name": "claude-instant-v1",
    "params": {
        "max_tokens_to_sample": 512,
        "temperature": 0.1
    }
}
```

## Huggingface [TODO]
To use models from [Huggingface](https://huggingface.co/), you can set `provider` to `huggingface_pipeline` when creating a labeling configuration. The specific model that will be queried can be specified using the `name` key. Autolabel currently supports all Sequence2Sequence Language Models on Huggingface. All models available on Huggingface can be found [here](https://huggingface.co/docs/transformers/model_doc/openai-gpt#:~:text=TEXT-,MODELS,-ALBERT). Ensure that the model you choose can be loaded using `AutoModelForSeq2SeqLM`. Here are a few examples:

* `google/flan-t5-small` (all flan-t5-* models)
* `google/pegasus-x-base`
* `microsoft/prophetnet-large-uncased`

This will run the model locally on a GPU (if available). You can also specify  quantization strategy to load larger models in lower precision (and thus decreasing memory requirements).

### Setup
To use Huggingface models with Autolabel, make sure to first install the relevant packages by running:
```bash
pip install refuel-autolabel[huggingface]
```

### Example usage
Here is an example of setting config to a dictionary that will use `google/flan-t5-small` model for labeling via Huggingface. Specifically, note that in the dictionary proivded by the `model` tag, `provider` is set to `huggingface_pipeline` and `name` is set to be `google/flan-t5-small`. `name` can be switched to use any model that satisfies the constraints above.

```python
config = {
    "task_name": "OpenbookQAWikipedia",
    "task_type": "multi_choice_question_answering",
    "dataset": {
        "label_column": "answer",
        "delimiter": ","
    },
    "model": {
        "provider": "huggingface_pipeline",
        "name": "google/flan-t5-small",
        "params": {}
    },
    "prompt": {
        "task_guidelines": "You are an expert at answering questions."
        "example_template": "Question: {question}\nAnswer: {answer}"
    }
}
```

### Additional parameters
A few parameters that can be passed in for `huggingface_pipeline` models to control the model behavior:

* `max_new_tokens` (int) - The maximum tokens to sample from the model
* `temperature` (float) - A float b/w 0 and 1 which indicates the diversity you want in the output. 0 uses greedy sampling.
* `quantize` (int) - The model quantization to use. 32 bit by default, but we also support 16 bit and 8 bit support for models which have been hosted on huggingface.

These parameters can be passed in via the `params` dictionary under `model`. Here is an example:
```python
"model": {
    "provider": "huggingface_pipeline",
    "name": "google/flan-t5-small",
    "params": {
        "max_new_tokens": 512,
        "temperature": 0.1,
        "quantize": 8
    }
},
```


## Google PaLM [TODO]

## Provider List
The table lists out all the provider, model combinations that Autolabel supports today:

| Provider     | Name              |
| -------------| ----------------- |
| openai       | text-davinci-003  |
| openai       | gpt-3.5-turbo     |
| openai       | gpt-4             |
| anthropic    | claude-v1         |
| anthropic    | claude-instant-v1 |
| huggingface_pipeline    | [seq2seq models](https://huggingface.co/learn/nlp-course/chapter1/7?fw=pt#sequencetosequence-modelssequencetosequencemodels) |
| refuel    | flan-t5-xxl |
