## Introduction

Text classification is a fundamental task in natural language processing (NLP) that involves categorizing textual data into predefined classes or categories. It is employed in various applications such as sentiment analysis, spam detection, topic classification, intent recognition, and document categorization and can be used in any setting where there are well defined categories which the LLM can understand and put an input into.

## Example [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x_CTOBG8uKV6O4wxsqWfaBL6G88szrDM)

### Dataset

Lets walk through using Autolabel for text classification on the Banking77 dataset. The Banking77 dataset comprises of 13,083 customer service queries labeled with 77 intents. It focuses on fine-grained single-domain intent detection. Every datapoint consists of an example and its corresponding label as shown below. The label belongs to a set of 77 predefined intents that the customer had for the particular datapoint for eg. activate_my_card, card_delivery_estimate, get_physical_card.

```json
{
    "example": "What can I do if my card still hasn't arrived after 2 weeks?",
    "label": "card_arrival"
}
```

Thus the dataset consists of just two columns, example and label. Here, Autolabel would be given the example input for a new datapoint and told to predict the label column which in this case is label.

### Config

In order to run Autolabel, we need a config defining the 3 important things - task, llm and dataset. Let's assume gpt-3.5-turbo as the LLM for this section.

```json
config = {
    "task_name": "BankingClassification",
    "task_type": "classification",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "prompt": {
        "task_guidelines": """You are an expert at understanding banking transaction complaints.\nYour job is to correctly label the provided input example into one of the following {num_labels} categories:\n{labels}""",
        "output_guidelines": "You will just return one line consisting of the label for the given example.",
        "labels": [
            "activate_my_card",
            "age_limit",
            "apple_pay_or_google_pay",
            ...
        ],
        "example_template": "Example: {example}\nOutput: {label}"
    }
}
```
The `task_type` sets up the config for a specific task, classification in this case.

Take a look at the prompt section of the config. This defines the settings related to defining the task and the machinery around it.  

The `task_guidelines` key is the most important key, it defines the task for the LLM to understand and execute on. In this case, we first set up the task and tell the model the kind of data present in the dataset, by telling it that it is an expert at understanding banking transaction complaints. Next, we define the task more concretely using the num_labels and labels appropriately. `{num_labels}` will be internally translated by the library to be the number of elements in the `labels` list (defined below).  `{labels}` will be translated to be all the labels in the `labels` list separated by a newline. These are essential for setting up classification tasks by telling it the labels that it is constrained to, along with any meaning associated with a label.  

The `labels` key defines the list of possible labels for the banking77 dataset which is a list of 77 possible labels.  

The `example_template` is one of the most important keys to set for a task. This defines the format of every example that will be sent to the LLM. This creates a prompt using the columns from the input dataset, and sends this prompt to the LLM hoping for the llm to generate the column defined under the `label_column`, which is label in our case. For every input, the model will be given the example with all the columns from the datapoint filled in according to the specification in the `example_template`. The `label_column` will be empty, and the LLM will generate the label. The `example_template` will be used to format all seed examples.  

### Few Shot Config

Let's assume we have access to a dataset of labeled seed examples. Here is a config which details how to use it.

```json
config = {
    "task_name": "BankingClassification",
    "task_type": "classification",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "prompt": {
        "task_guidelines": """You are an expert at understanding banking transaction complaints.\nYour job is to correctly label the provided input example into one of the following {num_labels} categories:\n{labels}""",
        "output_guidelines": "You will just return one line consisting of the label for the given example.",
        "labels": [
            "activate_my_card",
            "age_limit",
            "apple_pay_or_google_pay",
            ...
        ],
        "few_shot_examples": "../examples/banking/seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 5,
        "example_template": "Example: {example}\nOutput: {label}"
    }
}
```

The `few_shot_examples` key defines the seed set of labeled examples that are present for the model to learn from. A subset of these examples will be picked while querying the LLM in order to help it understand the task better, and understand corner cases.  

For the banking dataset, we found `semantic_similarity` search to work really well. This looks for examples similar to a query example from the seed set and sends those to the LLM when querying for a particular input. This is defined in the `few_shot_selection` key.  

`few_shot_num` defines the number of examples selected from the seed set and sent to the LLM. Experiment with this number based on the input token budget and performance degradation with longer inputs.

### Run the task

```py
from autolabel import LabelingAgent
agent = LabelingAgent(config)
ds = AutolabelDataset('data/banking77.csv', config = config)
agent.plan(ds)
agent.run(ds, max_items = 100)
```

### Evaluation metrics

On running the above config, this is an example output expected for labeling 100 items.
```
Cost in $=0.00, support=50, threshold=-inf, accuracy=0.6600, completion_rate=1.0000
Actual Cost: 0.0058579999999999995
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ support ┃ threshold ┃ accuracy ┃ completion_rate ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 100     │ -inf      │ 0.76     │ 1.0             │
└─────────┴───────────┴──────────┴─────────────────┘
```

**Accuracy** - We use accuracy as the main metric for evaluating classification tasks. This is done by checking the fraction of examples which are given the correct label in the training dataset.

**Completion Rate** - There can be errors while running the LLM related to labeling for eg. the LLM may give a label which is not in the label list or provide an answer which is not parsable by the library. In this cases, we mark the example as not labeled successfully. The completion rate refers to the proportion of examples that were labeled successfully.

### Notebook
You can find a Jupyter notebook with code that you can run on your own [here](https://github.com/refuel-ai/autolabel/blob/main/examples/banking/example_banking.ipynb)

## Classification Tasks with a Large Number of Classes

For classification tasks with a wide variety of possible classes, it is beneficial to run autolabel with `label_selection` turned on. In this mode, Autolabel will prune the list of possible classes to only include those that are similar to the example being labeled. This not only helps improve accuracy, but also substantially reduces labeling costs, as the size of the prompt decreases when classes are pruned.

To enable label_selection, simply set `label_selection` to `true` in your config file. Similarly, you can choose how many classes to select in the similarity search by setting `label_selection_count` to a value of your choosing.

```json
    "label_selection": true,
    "label_selection_count": 10
```

In this example, the list of classes will be reduced to only the 10 classes most similar to the example being labeled.

```json
config = {
    "task_name": "BankingClassification",
    "task_type": "classification",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "prompt": {
        "task_guidelines": """You are an expert at understanding banking transaction complaints.\nYour job is to correctly label the provided input example into one of the following {num_labels} categories:\n{labels}""",
        "output_guidelines": "You will just return one line consisting of the label for the given example.",
        "labels": [
            "activate_my_card",
            "age_limit",
            "apple_pay_or_google_pay",
            ...
        ],
        "few_shot_examples": "../examples/banking/seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 5,
        "example_template": "Example: {example}\nOutput: {label}",
        "label_selection": true,
        "label_selection_count": 10
    }
}
```
