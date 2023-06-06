# Large Language Models (LLMs)

Autolabel supports for multiple LLMs for labeling data. Some LLMs are available by calling an API with the appropriate API keys (OpenAI, Anthropic, etc.) while others can be run locally (such as the ones available on Huggingface).

Each LLMs belongs to an LLM provider -- which refers to the organization or open-source framework through which we are able to access the LLM. A full list of LLM providers and LLMs that are currently supported is provided below:

| Provider     | Name              |
| -------------| ----------------- |
| openai       | text-davinci-003  |
| openai       | gpt-3.5-turbo     |
| openai       | gpt-4             |

[needs to be filled out with all LLMs]

Autolabel makes it easy to try out different LLMs for your task and this page will walk you through how to get started with each LLM provider and model. Separately, we've also benchmarked multiple LLMs across different datasets - you can read the full technical report here [link to blog post] or check out the latest benchmark results here [link to benchmarks]. 


## OpenAI
To use models from [OpenAI](https://platform.openai.com/docs/models), you have to set the `provider` to `openai` when creating a labeling configuration. Autolabel currently supports the following models from OpenAI:

* `text-davinci-003`
* `gpt-3.5-turbo`
* `gpt-4`

`gpt-4` is the most capable (and most expensive) model from OpenAI, while `gpt-3.5-turbo` is the cheapest (but still quite capable). Pricing for these models is available [here](https://openai.com/pricing). 

### Installation [TODO]
To use OpenAI models with Autolabel, make sure to install the relevant packages by running:
```bash
pip install ...
```

### Example usage [TODO]

### Additional parameters
A few parameters that can be passed in for `openai` models:

* `max_tokens` (int): The maximum tokens to sample from the model
* `temperature` (float): A float between 0 and 2 which indicates the diversity you want in the output. 0 uses greedy sampling (picks the most likely outcome).

## Anthropic [TODO]
The provider_name here is 'anthropic'. The models supported are `claude-v1`, `claude-instant-v1`.

A few parameters that can be passed in for anthopic models:
max_tokens_to_sample (int) - The maximum tokens to sample from the model
temperature (float) - A float b/w 0 and 1 which indicates the diversity you want in the output. 0 uses greedy sampling.

## Huggingface [TODO]
The provider_name here is huggingface_pipeline. Any model available on the huggingface hub can be passed in as the model name. This runs the model locally on a GPU if that is available. Some models may be too big to fit on a GPU available locally to users, and so you can specify quantization strategy which makes the model smaller in terms of memory.

A few parameters that can be passed in for huggingface_pipeline models:
max_new_tokens (int) - The maximum tokens to sample from the model
temperature (float) - A float b/w 0 and 1 which indicates the diversity you want in the output. 0 uses greedy sampling.
quantize (int) - The model quantization to use. 32 bit by default, but we also support 16 bit and 8 bit support for models which have been hosted on huggingface.

## Google PaLM [TODO]
