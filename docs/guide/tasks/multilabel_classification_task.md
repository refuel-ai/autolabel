## Introduction

Multilabel text classification is a fundamental task in natural language processing (NLP) where textual data is categorized into predefined classes or categories. It expands upon traditional text classification by assigning multiple labels to each text instance. This approach finds applications in sentiment analysis, spam detection, topic classification, intent recognition, and document categorization. By considering multiple labels, it allows for a more nuanced representation of text data, accommodating scenarios where multiple topics or attributes are associated with a document. Multilabel text classification enables a flexible and comprehensive approach to categorizing textual data, providing a richer understanding of content and facilitating more nuanced decision-making in various NLP applications.

## Example [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1so1yjErzejgGXzNxUAgCNxSYPtI2Rl6E)

### Dataset

Lets walk through using Autolabel for multilabel text classification on the [sem_eval_2018_task_1 dataset](https://huggingface.co/datasets/sem_eval_2018_task_1) which we call twitter-emotion-detection for clarity. The twitter-emotion-detection dataset comprises of 10,983 English tweets and 11 emotions. If no emotions were selected for a row, we classified it as `neutral`.

```json
{
  "example": "I blew that opportunity -__- #mad",
  "label": "anger, disgust, sadness"
}
```

Thus the dataset consists of just two columns, example and labels. Here, Autolabel would be given the example input for a new datapoint and told to predict the label column which in this case is labels.

### Config

In order to run Autolabel, we need a config defining the 3 important things - task, llm and dataset. Let's assume gpt-3.5-turbo as the LLM for this section.

```json
config = {
    "task_name": "EmotionClassification",
    "task_type": "multilabel_classification",
    "dataset": {
        "label_column": "labels",
        "label_separator": ", ",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "prompt": {
        "task_guidelines": "You are an expert at classifying tweets as neutral or one or more of the given emotions that best represent the mental state of the poster.\nYour job is to correctly label the provided input example into one or more of the following categories:\n{labels}",
        "output_guidelines": "You will return the answer as a comma separated list of labels sorted in alphabetical order. For example: \"label1, label2, label3\"",
        "labels": [
            "neutral",
            "anger",
            "anticipation",
            ...
        ],
        "example_template": "Input: {example}\nOutput: {labels}"
    }
}
```

The `task_type` sets up the config for a specific task, multilabel_classification in this case.

Take a look at the prompt section of the config. This defines the settings related to defining the task and the machinery around it.

The `task_guidelines` key is the most important key, it defines the task for the LLM to understand and execute on. In this case, we first set up the task and tell the model the kind of data present in the dataset, by telling it that it is an expert at classifying tweets. Next, we define the task more concretely using labels appropriately. `{labels}` will be translated to be all the labels in the `labels` list separated by a newline. These are essential for setting up classification tasks by telling it the labels that it is constrained to, along with any meaning associated with a label.

The `labels` key defines the list of possible labels for the twitter-emotion-detection dataset which is a list of 12 possible labels.

The `example_template` is one of the most important keys to set for a task. This defines the format of every example that will be sent to the LLM. This creates a prompt using the columns from the input dataset, and sends this prompt to the LLM hoping for the llm to generate the column defined under the `label_column`, which is labels in our case. For every input, the model will be given the example with all the columns from the datapoint filled in according to the specification in the `example_template`. The `label_column` will be empty, and the LLM will generate the labels. The `example_template` will be used to format all seed examples.

### Few Shot Config

Let's assume we have access to a dataset of labeled seed examples. Here is a config which details how to use it.

```json
config = {
    "task_name": "EmotionClassification",
    "task_type": "multilabel_classification",
    "dataset": {
        "label_column": "labels",
        "label_separator": ", ",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "prompt": {
        "task_guidelines": "You are an expert at classifying tweets as neutral or one or more of the given emotions that best represent the mental state of the poster.\nYour job is to correctly label the provided input example into one or more of the following categories:\n{labels}",
        "output_guidelines": "You will return the answer as a comma separated list of labels sorted in alphabetical order. For example: \"label1, label2, label3\"",
        "labels": [
            "neutral",
            "anger",
            "anticipation",
            ...
        ],
        "few_shot_examples": "seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 5,
        "example_template": "Input: {example}\nOutput: {labels}"
    }
}

```

The `few_shot_examples` key defines the seed set of labeled examples that are present for the model to learn from. A subset of these examples will be picked while querying the LLM in order to help it understand the task better, and understand corner cases.

For the twitter dataset, we found `semantic_similarity` search to work really well. This looks for examples similar to a query example from the seed set and sends those to the LLM when querying for a particular input. This is defined in the `few_shot_selection` key.

`few_shot_num` defines the number of examples selected from the seed set and sent to the LLM. Experiment with this number based on the input token budget and performance degradation with longer inputs.

### Run the task

```py
from autolabel import LabelingAgent
agent = LabelingAgent(config)
agent.plan('twitter_emotion_detection.csv')
agent.run('twitter_emotion_detection.csv', max_items = 100)
```

### Evaluation metrics

On running the above config, this is an example output expected for labeling 100 items.

```
Actual Cost: 0.0025
┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ f1     ┃ support ┃ accuracy ┃ completion_rate ┃
┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 0.4507 │ 100     │ 0.08     │ 1.0             │
└────────┴─────────┴──────────┴─────────────────┘
```

**Accuracy** - This is calculated by taking the exact match of the predicted tokens and their correct class. This may suffer from class imbalance.

**F1** - This is calculated using the precision and recall of the predicted tokens and their classes. We use a macro average to get to one F1 score for all classes.

**Completion Rate** - There can be errors while running the LLM related to labeling for eg. the LLM may give a label which is not in the label list or provide an answer which is not parsable by the library. In this cases, we mark the example as not labeled successfully. The completion rate refers to the proportion of examples that were labeled successfully.

### Notebook

You can find a Jupyter notebook with code that you can run on your own [here](https://github.com/refuel-ai/autolabel/blob/main/examples/twitter_emotion_detection/example_twitter_emotion_detection.ipynb)
