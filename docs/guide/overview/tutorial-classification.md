This is a detailed tutorial that walks you through many features of the Autolabel library while solving a problem faced by many companies - labeling toxic comments for content moderation. We will be using OpenAI's `gpt-3.5-turbo` for the data labeling, and Refuel's LLM for confidence estimation.

If you want to run this code as you follow along, check out this Colab notebook: [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1t-9vNLkyoyySAG_0w3eR98biBOXlMO-E?usp=sharing)


## Autolabel installation

Since we'll be using OpenAI along with Autolabel, we can install all necessary libraries by simply running:
```bash
pip install 'refuel-autolabel[openai]'
```

Now, we can set our OpenAI key as an environment variable to get started. You can always use an LLM of your choice - see more optioons and installation instructions [here](/guide/llms/llms). 

## Download and review dataset

We'll be using a dataset called [Civil Comments](https://huggingface.co/datasets/civil_comments), which is [available through Autolabel](/guide/resources/refuel_datasets). You can download it locally, by simply running:
```python
from autolabel import get_data

get_data('civil_comments')
```

The output is:
```
Downloading seed example dataset to "seed.csv"...
100% [..............................................................................] 65757 / 65757
Downloading test dataset to "test.csv"...
100% [............................................................................] 610663 / 610663
```

This results in two files being downloaded locally:

* `seed.csv`: small dataset with labels that we'll rely on as helpful examples.
* `test.csv`: larger dataset that we are trying to label.

A few examples are shown below:

| label      | examples                                                                              |
| ---------- | ------------------------------------------------------------------------------------- |
| `toxic`    | "The ignorance and bigotry comes from your post!"                                     |
| `not toxic`| "This is malfeasance by the Administrator and the Board. They are wasting our money!" |

## Start the labeling process
Labeling with Autolabel is a 3-step process:

* First, we specify a labeling configuration (see `config` object below) and create a `LabelingAgent`
* Next, we do a dry-run on our dataset using the LLM specified in `config` by running `agent.plan`
* Finally, we run the labeling with `agent.run`

### Experiment #1: Try simple labeling guidelines

Define the configuration file below:
```python
config = {
    "task_name": "ToxicCommentClassification",
    "task_type": "classification", # classification task
    "dataset": {
        "label_column": "label",
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo" # the model we want to use
    },
    "prompt": {
        # very simple instructions for the LLM
        "task_guidelines": "Does the provided comment contain 'toxic' language? Say toxic or not toxic.",
        "labels": [ # list of labels to choose from
            "toxic",
            "not toxic"
        ],
        "example_template": "Input: {example}\nOutput: {label}"
    }
}
```

Now, we do the dry-run with `agent.plan`:
```python
from autolabel import LabelingAgent

agent = LabelingAgent(config)
agent.plan('test.csv')
```

Output:
```console
┌──────────────────────────┬─────────┐
│ Total Estimated Cost     │ $4.4442 │
│ Number of Examples       │ 2000    │
│ Average cost per example │ $0.0022 │
└──────────────────────────┴─────────┘
───────────────────────────────────────────────── Prompt Example ──────────────────────────────────────────────────
Does the provided comment contain 'toxic' language? Say toxic or not toxic.

You will return the answer with just one element: "the correct label"

Now I want you to label the following example:
Input: [ Integrity means that you pay your debts.]. Does this apply to President Trump too?
Output: 

```

Finally, we run the data labeling:
```python
labels, df, metrics = agent.run('test.csv', max_items=100)
```

```
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ support ┃ threshold ┃ accuracy ┃ completion_rate ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 100     │ -inf      │ 0.54     │ 1.0             │
└─────────┴───────────┴──────────┴─────────────────┘
```

54% accuracy is not very good! Let's see if we can improve this further!

### Experiment #2: Few-shot prompting to provide helpful examples

Similar to how human labelers find it helpful to use relevant examples when making a decision, LLM performance for labeling also goes up when choosing helpful examples in the prompt. For this next experiment, we will pick a few helpful examples from `seed.csv`. More information on few-shot prompting can be found [here](/guide/accuracy/few-shot).

We take the previous config, and just update the following fields:
```python
{
    "task_name": "ToxicCommentClassification",
    "task_type": "classification",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo",
    },
    "prompt": {
        "task_guidelines":  "Does the provided comment contain 'toxic' language? Say toxic or not toxic.",
        "labels": [
            "toxic",
            "not toxic"
        ],
        "few_shot_examples": [
            {
                "example": "It's ridiculous that these guys are being called 'protesters'. Being armed is a threat of violence, which makes them terrorists.",
                "label": "toxic"
            },
            {
                "example": "This is so cool. It's like, 'would you want your mother to read this??' Really great idea, well done!",
                "label": "not toxic"
            },
            {
                "example": "This bitch is nuts. Who would read a book by a woman",
                "label": "toxic"
            },
            {
                "example": "It was a great show. Not a combo I'd of expected to be good together but it was.",
                "label": "not toxic"
            }
        ],
        "few_shot_selection": "fixed",
        "few_shot_num": 4,
        "example_template": "Input: {example}\nOutput: {label}"
    }
}
```

That's it! We are now ready to create a `LabelingAgent` and run the same `agent.plan` and `agent.run` commands.

```console
┌──────────────────────────┬─────────┐
│ Total Estimated Cost     │ $4.9442 │
│ Number of Examples       │ 2000    │
│ Average cost per example │ $0.0025 │
└──────────────────────────┴─────────┘
───────────────────────────────────────────────── Prompt Example ──────────────────────────────────────────────────
Does the provided comment contain 'toxic' language? Say toxic or not toxic.

You will return the answer with just one element: "the correct label"

Some examples with their output answers are provided below:

Input: It's ridiculous that these guys are being called 'protesters'. Being armed is a threat of violence, which makes them terrorists.
Output: toxic

Input: This is so cool. It's like, 'would you want your mother to read this??' Really great idea, well done!
Output: not toxic

Input: This bitch is nuts. Who would read a book by a woman
Output: toxic

Input: It was a great show. Not a combo I'd of expected to be good together but it was.
Output: not toxic

Now I want you to label the following example:
Input: [ Integrity means that you pay your debts.] Does this apply to President Trump too?
Output:
```

With additional examples, the cost has gone up slightly. Now, we run the labeling with:

```python
labels, df, metrics = agent.run('test.csv', max_items=100)`:
```

```console
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ support ┃ threshold ┃ accuracy ┃ completion_rate ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 100     │ -inf      │ 0.68     │ 1.0             │
└─────────┴───────────┴──────────┴─────────────────┘
```

Nice! We improved performance from 54% to 68% by providing a few examples to the LLM.

### Experiment #3: Improving task guidelines after reviewing errors (prompt engineering)

Typically, you can improve the accuracy by reviewing mistakes and updating the task guidelines (see another example [here](/guide/accuracy/prompting-better)). You can review some of the mistakes from the previous run by looking at the output Pandas DataFrame produced called `df`:
```python
df[df['label'] != df['ToxicCommentClassification_llm_label']].head(10)
```

Let's say we update our task guidelines to be more explicit about how should the LLM make the decision about whether a comment is toxic or not:

```python
{
    "task_name": "ToxicCommentClassification",
    "task_type": "classification",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo",
    },
    "prompt": {
        "task_guidelines": "You are an expert at identifying toxic comments. You aim to act in a fair and balanced manner, where comments that provide fair criticism of something or someone are labelled 'not toxic'. Similarly, criticisms of policy and politicians are marked 'not toxic', unless the comment includes obscenities, racial slurs or sexually explicit material. Any comments that are sexually explicit, obscene, or insults a person, demographic or race are not allowed and labeled 'toxic'. \nYour job is to correctly label the provided input example into one of the following categories:\n{labels}",
        "labels": [
            "toxic",
            "not toxic"
        ],
        "few_shot_examples": [
            {
                "example": "It's ridiculous that these guys are being called 'protesters'. Being armed is a threat of violence, which makes them terrorists.",
                "label": "toxic"
            },
            {
                "example": "This is so cool. It's like, 'would you want your mother to read this??' Really great idea, well done!",
                "label": "not toxic"
            },
            {
                "example": "This bitch is nuts. Who would read a book by a woman",
                "label": "toxic"
            },
            {
                "example": "It was a great show. Not a combo I'd of expected to be good together but it was.",
                "label": "not toxic"
            }
        ],
        "few_shot_selection": "fixed",
        "few_shot_num": 4,
        "example_template": "Input: {example}\nOutput: {label}"
    }
}
```

Now, when we run `agent.run`, we get the following results:

```
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ support ┃ threshold ┃ accuracy ┃ completion_rate ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 100     │ -inf      │ 0.78     │ 1.0             │
└─────────┴───────────┴──────────┴─────────────────┘
```

We now hit an accuracy of 78%, which is very promising! If we spend more time improving the guidelines or choosing different examples, we can push accuracy even further.

### Experiment #4: Experimenting with LLMs

We've iterated a fair bit on prompts, and few-shot examples. Let's evaluate a few different LLMs provided by the library out of the box. For example, we observe that we can boost performance even further by using `text-davinci-003`: 

```python
{
    "task_name": "ToxicCommentClassification",
    "task_type": "classification",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "text-davinci-003",
    },
    "prompt": {
        "task_guidelines": "You are an expert at identifying toxic comments. You aim to act in a fair and balanced manner, where comments that provide fair criticism of something or someone are labelled 'not toxic'. Similarly, criticisms of policy and politicians are marked 'not toxic', unless the comment includes obscenities, racial slurs or sexually explicit material. Any comments that are sexually explicit, obscene, or insults a person, demographic or race are not allowed and labeled 'toxic'. \nYour job is to correctly label the provided input example into one of the following categories:\n{labels}",
        "labels": [
            "toxic",
            "not toxic"
        ],
        "few_shot_examples": [
            {
                "example": "It's ridiculous that these guys are being called 'protesters'. Being armed is a threat of violence, which makes them terrorists.",
                "label": "toxic"
            },
            {
                "example": "This is so cool. It's like, 'would you want your mother to read this??' Really great idea, well done!",
                "label": "not toxic"
            },
            {
                "example": "This bitch is nuts. Who would read a book by a woman",
                "label": "toxic"
            },
            {
                "example": "It was a great show. Not a combo I'd of expected to be good together but it was.",
                "label": "not toxic"
            }
        ],
        "few_shot_selection": "fixed",
        "few_shot_num": 4,
        "example_template": "Input: {example}\nOutput: {label}"
    }
}
```

While the per token API price for this model is higher, we're able to boost the accuracy to 88%!

```console
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ support ┃ threshold ┃ accuracy ┃ completion_rate ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 100     │ -inf      │ 0.88     │ 1.0             │
└─────────┴───────────┴──────────┴─────────────────┘
```

### Experiment #5: Using confidence scores

Refuel provides LLMs that can compute confidence scores for every label, if the LLM you've chosen doesn't provide token-level log probabilities. This is helpful, because you can calibrate a confidence threshold for your labeling task, and then route less confident labels to humans, while you still get the benefits of auto-labeling for the confident examples. Let's see how this works. 

First, set your Refuel API key as an environment variable (and if you don't have this key yet, sign up <a href="https://refuel-ai.typeform.com/llm-access" target="_blank">here</a>).
```python
os.environ['REFUEL_API_KEY'] = '<your-api-key>'
```

Now, update your configuration:
```python
config["model"]["compute_confidence"] = True
```

Finally, let's run `agent.run` as before - this produces the table below:
```
Metric: auroc: 0.8858
Actual Cost: 0.0376
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ support ┃ threshold ┃ accuracy ┃ completion_rate ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 100     │ -inf      │ 0.78     │ 1.0             │
│ 1       │ 0.9988    │ 1.0      │ 0.01            │
│ 12      │ 0.9957    │ 1.0      │ 0.12            │
│ 13      │ 0.9949    │ 0.9231   │ 0.13            │
│ 54      │ 0.9128    │ 0.9815   │ 0.54            │
│ 55      │ 0.9107    │ 0.9636   │ 0.55            │
│ 63      │ 0.6682    │ 0.9683   │ 0.63            │
│ 66      │ 0.6674    │ 0.9242   │ 0.66            │
│ 67      │ 0.6673    │ 0.9254   │ 0.67            │
│ 69      │ 0.6671    │ 0.8986   │ 0.69            │
│ 71      │ 0.6667    │ 0.9014   │ 0.71            │
│ 72      │ 0.6667    │ 0.8889   │ 0.72            │
│ 78      │ 0.4819    │ 0.8974   │ 0.78            │
│ 79      │ 0.4774    │ 0.8861   │ 0.79            │
│ 87      │ 0.4423    │ 0.8966   │ 0.87            │
│ 100     │ 0.0402    │ 0.78     │ 1.0             │
└─────────┴───────────┴──────────┴─────────────────┘
```

The rows in this table show labeling performance at different confidence thresholds, and set an autolabeling confidence threshold at the desired accuracy. For instance, from the table above we can set the confidence threshold at 0.6682 which allows us to label at 96% accuracy with a completion rate of 63%.

If you want to run this code as you follow along, check out this Colab notebook: [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1t-9vNLkyoyySAG_0w3eR98biBOXlMO-E?usp=sharing)

## Final thoughts

Hopefully, this tutorial was helpful in understanding how Autolabel can help you label datasets quickly and at high quality. A Jupyter notebook for this tutorial can be found [here](https://github.com/refuel-ai/autolabel/blob/main/examples/civil_comments/example_civil_comments.ipynb).

You can find more example notebooks [here](https://github.com/refuel-ai/autolabel/tree/main/examples), including for tasks such as question answering, named entity recognition, etc. 

Drop us a message in our <a href="https://discord.gg/uEdr8nrMGm" target="_blank">Discord</a> if you want to chat with us, or go to [Github](https://github.com/refuel-ai/autolabel/issues) to report any issues!