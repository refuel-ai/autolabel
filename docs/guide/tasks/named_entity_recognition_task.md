## Introduction

Named Entity Recognition (NER) is a crucial task in natural language processing (NLP) that involves identifying and classifying named entities in text. Named entities refer to specific individuals, organizations, locations, dates, quantities, and other named entities present in the text. The goal of NER is to extract and classify these entities accurately, providing valuable information for various NLP applications such as information extraction, question answering, and sentiment analysis.

## Example [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M87KAnjf0KEtAI69BnYc_pwsjfkTvMrK#scrollTo=c93fae0b)

### Dataset

Lets walk through using Autolabel for named entity recognition on the CONLL2003 dataset. The CONLL2003 dataset comprises of sentences with entities in the sentence labeled LOC (location), ORG (organization), PER (person) or MISC (Miscellaneous).  

```json
{
    "example": "The role of the 70,000 mainly Kurdish village guards who fight Kurdistan Workers Party ( PKK ) guerrillas in the southeast has been questioned recently after media allegations that many of them are involved in common crime .",
    "CategorizedLabels": "{'Location': [], 'Organization': ['Kurdistan Workers Party', 'PKK'], 'Person': [], 'Miscellaneous': ['Kurdish']}"
}
```

Thus the dataset consists of the `example` and `CategorizedLabels` columns. Here `example` mentions the sentence which needs to be labeled. The `CategorizedLabels` contains the entities for every label as a list.

### Config

In order to run Autolabel, we need a config defining the 3 important things - task, llm and dataset. Let's assume gpt-3.5-turbo as the LLM for this section.

```py
config = {
    "task_name": "PersonLocationOrgMiscNER",
    "task_type": "named_entity_recognition",
    "dataset": {
        "label_column": "CategorizedLabels",
        "text_column": "example"
    },
    "model": {
        "provider": "anthropic",
        "name": "claude-v1"
    },
    "prompt": {
        "task_guidelines": "You are an expert at extracting Person, Organization, Location, and Miscellaneous entities from text. Your job is to extract named entities mentioned in text, and classify them into one of the following categories.\nCategories:\n{labels}\n ",
        "labels": [
            "Location",
            "Organization",
            "Person",
            "Miscellaneous"
        ],
        "example_template": "Example: {example}\nOutput: {CategorizedLabels}",
        "few_shot_examples": "data/conll2003_seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 5
    }
}
```
The `task_type` sets up the config for a specific task, named_entity_recognition in this case.

Take a look at the prompt section of the config. This defines the settings related to defining the task and the machinery around it.  

The `task_guidelines` key is the most important key, it defines the task for the LLM to understand and execute on. In this case, we first set up the task and tell the model the kind of data present in the dataset, by telling it that it is an expert at extracting entities from text and classifying them into the necessary labels. Next, we tell the model the list of categories that it should classify every entity into. This ensures that every entity is assigned to one category.  

The `example_template` is one of the most important keys to set for a task. This defines the format of every example that will be sent to the LLM. This creates a prompt using the columns from the input dataset, and sends this prompt to the LLM hoping for the llm to generate the column defined under the `label_column`, which is `CategorizedLabels` in our case. For every input, the model will be given the example with all the columns from the datapoint filled in according to the specification in the `example_template`. The `label_column` will be empty, and the LLM will generate the label. The `example_template` will be used to format all seed examples.  

The `few_shot_examples` here is a path to a csv which defines a set of labeled examples which the model can use to understand the task better. These examples will be used as a reference by the model.

`few_shot_num` defines the number of examples selected from the seed set and sent to the LLM. Experiment with this number based on the input token budget and performance degradation with longer inputs.

`few_shot_selection` is set to `semantic_similarity` in this case as we want to use a subset of examples as seed examples from a larger set to get dynamically good seed examples.

### Run the task

```py
from autolabel import LabelingAgent
agent = LabelingAgent(config)
agent.plan('examples/squad_v2/test.csv', max_items = 100)
agent.run('examples/squad_v2/test.csv', max_items = 100)
```

### Evaluation metrics

On running the above config, this is an example output expected for labeling 100 items.
```
┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ f1     ┃ support ┃ threshold ┃ accuracy ┃ completion_rate ┃
┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 0.7834 │ 100     │ -inf      │ 0.7834   │ 1.0             │
└────────┴─────────┴───────────┴──────────┴─────────────────┘
```

**Accuracy** - This is calculated by taking the exact match of the predicted tokens and their correct class. This may suffer from class imbalance.

**F1** - This is calculated using the precision and recall of the predicted tokens and their classes. We use a macro average to get to one F1 score for all classes.

**Completion Rate** - There can be errors while running the LLM related to labeling for eg. the LLM may provide an answer which is not parsable by the library. In this cases, we mark the example as not labeled successfully. The completion rate refers to the proportion of examples that were labeled successfully.
