# Embedding Models

Autolabel also supports various models to compute text embeddings that are used in some few shot example selection strategies such as [semantic similarity and max marginal relevance](/guide/accuracy/few-shot). Like the LLMs that Autolabel supports, each embedding model belongs to a provider. Currently the library supports embedding models from 3 providers: OpenAI, Google Vertex AI, and Huggingface. By default, if no embedding config is present in the labeling config but a few shot strategy that requires text embeddings is enabled, Autolabel defaults to use OpenAI embeddings and an OpenAI API key will be required. 

Details on how to set up the embedding config for each provider are below.

## OpenAI
To use models from [OpenAI](https://platform.openai.com/docs/models), you can set `provider` to `openai` under the `embedding` key in the labeling configuration. Then, the specific model that will be queried can be specified using the `model` key. The default embedding model, if none is provided, is `text-embedding-ada-002`

### Setup
To use OpenAI models with Autolabel, make sure to first install the relevant packages by running:
```bash
pip install 'refuel-autolabel[openai]'
```
and also setting the following environment variable:
```
export OPENAI_API_KEY=<your-openai-key>
```
replacing `<your-openai-key>` with your API key, which you can get from [here](https://platform.openai.com/account/api-keys).

### Example usage
Here is an example of setting config to a dictionary that will use OpenAI's `text-embedding-ada-002` model for computing text embeddings. Specifically, note that in the dictionary provided by the `embedding` tag, `provider` is set to `openai` and `model` is not set so it will default to `text-embedding-ada-002`.

```python
config = {
    "task_name": "OpenbookQAWikipedia",
    "task_type": "question_answering",
    "dataset": {
        "label_column": "answer",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo",
        "params": {}
    },
    "embedding": {
        "provider": "openai"
    },
    "prompt": {
        "task_guidelines": "You are an expert at answering questions.",
        "example_template": "Question: {question}\nAnswer: {answer}"
    }
}
```

## Hugging Face
To use models from [Hugging Face](https://huggingface.co/), you can set `provider` to `huggingface_pipeline` when creating a labeling configuration. The specific model that will be queried can be specified using the `name` key. 

This will run the model locally on a GPU (if available). You can also specify  quantization strategy to load larger models in lower precision (and thus decreasing memory requirements).

### Setup
To use Hugging Face models with Autolabel, make sure to first install the relevant packages by running:
```bash
pip install 'refuel-autolabel[huggingface]'
```

### Example usage
Here is an example of setting config to a dictionary that will use the `sentence-transformers/all-mpnet-base-v2` model for computing text embeddings. Specifically, note that in the dictionary provided by the `embedding` tag, `provider` is set to `huggingface_pipeline` and `model` is set to be `sentence-transformers/all-mpnet-base-v2`.

```python
config = {
    "task_name": "OpenbookQAWikipedia",
    "task_type": "question_answering",
    "dataset": {
        "label_column": "answer",
        "delimiter": ","
    },
    "model": {
        "provider": "huggingface_pipeline",
        "name": "google/flan-t5-small",
        "params": {}
    },
    "embedding": {
        "provider": "huggingface_pipeline",
        "model": "sentence-transformers/all-mpnet-base-v2"
    },
    "prompt": {
        "task_guidelines": "You are an expert at answering questions.",
        "example_template": "Question: {question}\nAnswer: {answer}"
    }
}
```

## Google Vertex AI
To use models from [Google](https://developers.generativeai.google/products/palm), you can set the `provider` to `google` when creating a labeling configuration. The specific model that will be queried can be specified using the `model` key. 

### Setup
To use Google models with Autolabel, make sure to first install the relevant packages by running:
```bash
pip install 'refuel-autolabel[google]'
```
and also setting up [Google authentication](https://cloud.google.com/docs/authentication/application-default-credentials) locally.

### Example usage
Here is an example of setting config to a dictionary that will use google's `textembedding-gecko` model for computing text embeddings. Specifically, note that in the dictionary provided by the `embedding` tag, `provider` is set to `google` and `model` is set to be `textembedding-gecko`.

```python
config = {
    "task_name": "OpenbookQAWikipedia",
    "task_type": "question_answering",
    "dataset": {
        "label_column": "answer",
        "delimiter": ","
    },
    "model": {
        "provider": "google",
        "name": "text-bison@001",
        "params": {}
    },
    "embedding": {
        "provider": "google",
        "model": "textembedding-gecko"
    }
    "prompt": {
        "task_guidelines": "You are an expert at answering questions.",
        "example_template": "Question: {question}\nAnswer: {answer}"
    }
}
```
