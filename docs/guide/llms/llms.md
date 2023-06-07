# Large Language Models (LLMs)

Autolabel supports for multiple LLMs for labeling data. Some LLMs are available by calling an API with the appropriate API keys (OpenAI, Anthropic, etc.) while others can be run locally (such as the ones available on Huggingface).

Each LLMs belongs to an LLM provider -- which refers to the organization or open-source framework through which we are able to access the LLM. A full list of LLM providers and LLMs that are currently supported is provided below:

| Provider     | Name              |
| -------------| ----------------- |
| openai       | text-davinci-003  |
| openai       | gpt-3.5-turbo     |
| openai       | gpt-4             |
| google       | text-bison@001    |
| google       | chat-bison@001    |

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

## Google PaLM
To use models from [Google](https://developers.generativeai.google/products/palm), you can set the `provider` to `google` when creating a labeling configuration. The specific model that will be queried can be specified using the `name` key. Autolabel currently supports the following models from Google:

* `text-bison@001`
* `chat-bison@001`

`text-bison@001` is often more suitable for labeling tasks due to its ability to follow natural language instructions. `chat-bison@001` is fine-tuned for multi-turn conversations. `text-bison@001` costs $0.001/1K characters and `chat-bison@001` costs half that at $0.0005/1K characters. Detailed pricing for these models is available [here](https://cloud.google.com/vertex-ai/pricing#generative_ai_models)

### Setup
To use Google models with Autolabel, make sure to first install the relevant packages by running:
```bash
pip install refuel-autolabel[google]
```
and also setting up [Google authentication](https://cloud.google.com/docs/authentication/application-default-credentials) locally.

### Example usage
Here is an example of setting config to a dictionary that will use google's `text-bison@001` model for labeling. Specifically, note that in the dictionary provided by the `model` tag, `provider` is set to `google` and `name` is set to be `text-bison@001`. `name` can be switched to use any of the two models mentioned above.

```python
config = {
    "task_name": "OpenbookQAWikipedia",
    "task_type": "multi_choice_question_answering",
    "dataset": {
        "label_column": "answer",
        "delimiter": ","
    },
    "model": {
        "provider": "google",
        "name": "text-bison@001",
        "params": {}
    },
    "prompt": {
        "task_guidelines": "You are an expert at answering questions."
        "example_template": "Question: {question}\nAnswer: {answer}"
    }
}
```

### Additional parameters
A few parameters can be passed in alongside `google` models to tweak their behavior:

* `max_output_tokens` (int): Maximum number of tokens that can be generated in the response.
* `temperature` (float): A float between 0 and 1 which indicates the diversity you want in the output. 0 uses greedy sampling (picks the most likely outcome).

These parameters can be passed in via the `params` dictionary under `model`. Here is an example:
```python
"model": {
    "provider": "google",
    "name": "text-bison@001",
    "params": {
        "max_output_tokens": 512,
        "temperature": 0.1
    }
}
```

### Model behavior
`chat-bison@001` always responds in a "chatty" manner (example below), often returning much more than just the requested label. This can cause problems on certain labeling tasks.

Prompt
```
You are an expert at understanding emotions of human texts.
Your job is to correctly label the provided input example into one of the following categories:
sadness
joy
love
anger
fear
surprise

You will return the answer with just one element: "the correct label"

Some examples with their output answers are provided below:

Example: i am pinned as the culprit of digging out their inferiority and made them feel useless again
Output: sadness

Example: i feel about the plight of these dogs so its lovely to find a turkish vet who really cares
Output: love

Example: i see how strong and bright you are and as you meet your milestones weeks early i feel assured that my gut was always right
Output: joy

Now I want you to label the following example:
Example: im feeling rather rotten so im not very ambitious right now
Output:
```
Reponse from `chat-bison@001`
```
The speaker's lack of ambition is likely a result of their sadness and low energy. They may not feel like they have the motivation to do anything
```

### Content moderation
Both Google LLMs seem to have much stricter content moderation rules than the other supported models. Consider a different model if your dataset has content that is likely to trigger Google's built-in content moderation.