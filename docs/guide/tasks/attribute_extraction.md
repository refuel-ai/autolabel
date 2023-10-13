## Introduction

Attribute Extraction is a task that shows up in real world frequently. This task extracts multiple attributes or features from a single piece of text. For eg. extracting the colour, price and name from a product description paragraph. Instead of making multiple calls to the llm, we can extract all attributes in one call! Additionally, if the attributes are related to each other, doing attribute extraction means that the relationships between the outputs are respected i.e suppose we extract the length of a shirt along with its letter size. Doing attribute extraction would make sure the letter and the integer length are consistent.

## Example [![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12kyDbJltfrBW7WxKV38NOQVE-df6IOIT)

### Dataset

Lets walk through using Autolabel for attribute extraction on the ethos dataset. The ethos dataset comprises of hate speech on social media platforms. Every datapoints consists of an exmaple with hate speech and corresponding to it, there are three attributes, i.e violence, gender and directed_vs_generalized.

```json
{
  "example": "tweet containing hate speech",
  "violence": "violent",
  "directed_vs_generalized": "directed",
  "gender": "false"
}
```

Thus the dataset contains of 4 columns, the example along with the 3 attributes. Here, Autolabel would be given the example input for a new datapoint and told to predict the labels for the 3 attributes.

## Config

In order to run Autolabel, we need a config defining the 3 important things - task, llm and dataset. Let's assume gpt-3.5-turbo as the LLM for this section.

```json
config = {
  "task_name": "EthosAttributeExtraction",
  "task_type": "attribute_extraction",
  "dataset": {
    "text_column": "text",
    "delimiter": ","
  },
  "model": {
    "provider": "openai",
    "name": "gpt-3.5-turbo"
  },
  "prompt": {
    "task_guidelines": "You are an expert at classifying hate speech and identifying the type of hate speech. Read the following tweets and extract the following attributes from the text.",
    "attributes": [
      {
        "name": "violence",
        "options": ["not_violent", "violent"],
        "description": "If the tweet mentions violence towards a person or a group."
      },
      {
        "name": "directed_vs_generalized",
        "options": [
          "generalized",
          "directed"
        ],
        "description": "If the hate speech is generalized towards a group or directed towards a specific person."
      },
      {
        "name": "gender",
        "options": [
          "true",
          "false"
        ],
        "description": "If the hate speech uses gendered language and attacks a particular gender."
      }
    ],
    "few_shot_examples": "seed.csv",
    "few_shot_selection": "fixed",
    "few_shot_num": 5,
    "example_template": "Text: {text}\nOutput: {output_dict}"
  }
}
```

The `task_type` sets up the config for a specific task, attribute_extraction in this case.

Take a look at the prompt section of the config. This defines the settings related to defining the task and the machinery around it.

The `task_guidelines` key is the most important key, it defines the task for the LLM to understand and execute on. In this case, we first set up the task and tell the model the kind of data present in the dataset, by telling it that it is an expert at classifying hate speech.

The `attributes` key is the most important key for defining attribute extraction well. For every attribute, we have atleast 2 keys -  
 a. `name` - This is the name of the attribute.
b. `description` - This is the description of an attribute. This describes the attribute more concretely and prompts the model to extract the corresponding attribute.
c. `options` - You can also define a list of options for the LLM. This is an optional field. In case the attribute has a list of values from which to choose the value, fill this list. Otherwise, the attribute is prompted to be any possible textual value.

The `example_template` is one of the most important keys to set for a task. This defines the format of every example that will be sent to the LLM. This creates a prompt using the columns from the input dataset. Here we define the `output_dict` key, which is used in the example template for attribute extraction tasks. This will create a json of all the attributes, as key value pairs. The LLM is also prompted to output the attributes in a json.

### Run the task

```py
from autolabel import LabelingAgent, AutolabelDataset
agent = LabelingAgent(config)
ds = AutolabelDataset('test.csv', config = config)
agent.plan(ds)
agent.run(ds, max_items = 100)
```

### Evaluation metrics

On running the above config, this is an example output expected for labeling 100 items.

```
Actual Cost: 0.0665
┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ violence:… ┃ violence:… ┃ violence:… ┃ directed_… ┃ directed… ┃ directed_… ┃ gender:s… ┃ gender:co… ┃ gender:a… ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 100        │ 1.0        │ 0.89       │ 100        │ 1.0       │ 0.89       │ 100       │ 1.0        │ 0.94      │
└────────────┴────────────┴────────────┴────────────┴───────────┴────────────┴───────────┴────────────┴───────────┘
```

**Accuracy** - This is calculated by taking the exact match of the predicted tokens and their correct class. This may suffer from class imbalance.

**Completion Rate** - There can be errors while running the LLM related to labeling for eg. the LLM may give a label which is not in the label list or provide an answer which is not parsable by the library. In this cases, we mark the example as not labeled successfully. The completion rate refers to the proportion of examples that were labeled successfully.

### Confidence

You can calculate per attribute confidence metric as well by setting compute_confidence as true in the model config. This can help you decide which examples to keep per attribute.

### Notebook

You can find a Jupyter notebook with code that you can run on your own [here](https://github.com/refuel-ai/autolabel/blob/main/examples/ethos/example_ethos.ipynb).
