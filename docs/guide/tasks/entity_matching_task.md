## Introduction

Entity matching in natural language processing (NLP) is a task that involves identifying and matching entities from different sources or datasets based on various fields or attributes. The goal is to determine if two entities refer to the same real-world object or entity, even if they are described differently or come from different data sources.

## Example

### Dataset

Lets walk through using Autolabel for entity matching on the Walmart-Amazon dataset. This dataset consists of duplicate products listed on both Walmart and Amazon. These products would have different names and descriptions but would be the same product. The dataset consists of such examples, where given the name and the description, the task is to predict if the products are duplicate or not. An example from the Walmart-Amazon dataset,

```json
{
    "entity1": "Title: zotac geforce gt430 1gb ddr3 pci-express 2.0 graphics card; Category: electronics - general; Brand: zotac; ModelNo: zt-40604-10l; Price: 88.88;",
    "entity2": "Title: evga geforce gts450 superclocked 1 gb gddr5 pci-express 2.0 graphics card 01g-p3-1452-tr; Category: graphics cards; Brand: evga; ModelNo: 01g-p3-1452-tr; Price: 119.88;",
    "label": "not duplicate"
}
```

The the dataset consists of two columns `entity1` and `entity2` which define the two entities. There could also be multiple columns defining an entity. The `label` column here defines if the two entities are duplicates or not.

### Config

In order to run Autolabel, we need a config defining the 3 important things - task, llm and dataset. Let's assume gpt-3.5-turbo as the LLM for this section.

```json
config = {
    "task_name": "ProductCatalogEntityMatch",
    "task_type": "entity_matching",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo",
        "params": {}
    },
    "prompt": {
        "task_guidelines": "You are an expert at identifying duplicate products from online product catalogs.\nYou will be given information about two product entities, and your job is to tell if they are the same (duplicate) or different (not duplicate). Your answer must be from one of the following options:\n{labels}",
        "labels": [
            "duplicate",
            "not duplicate"
        ],
        "few_shot_examples": [
            {
                "entity1": "Title: lexmark extra high yield return pgm print cartridge - magenta; Category: printers; Brand: lexmark; ModelNo: c782u1mg; Price: 214.88;",
                "entity2": "Title: lexmark 18c1428 return program print cartridge black; Category: inkjet printer ink; Brand: lexmark; ModelNo: 18c1428; Price: 19.97;",
                "label": "not duplicate"
            },
            {
                "entity1": "Title: edge tech proshot 4gb sdhc class 6 memory card; Category: usb drives; Brand: edge tech; ModelNo: pe209780; Price: 10.88;",
                "entity2": "Title: 4gb edge proshot sdhc memory card class6; Category: computers accessories; Brand: edge; ModelNo: nan; Price: 17.83;",
                "label": "duplicate"
            },
            {
                "entity1": "Title: tomtom one carry case; Category: gps; Brand: tomtom; ModelNo: 9n00 .181; Price: 19.96;",
                "entity2": "Title: tomtom one carrying case; Category: cases; Brand: tomtom; ModelNo: 9n00 .181; Price: 4.99;",
                "label": "duplicate"
            },
            {
                "entity1": "Title: iosafe rugged 250gb usb 3.0 portable external hard drive; Category: hard drives; Brand: iosafe; ModelNo: pa50250u5yr; Price: 249.99;",
                "entity2": "Title: lacie rugged all-terrain 500 gb firewire 800 firewire 400 usb 2.0 portable external hard drive 301371; Category: external hard drives; Brand: lacie; ModelNo: 301371; Price: nan;",
                "label": "not duplicate"
            }
        ],
        "few_shot_selection": "fixed",
        "few_shot_num": 3,
        "example_template": "Entity1: {entity1}\nEntity2: {entity2}\nOutput: {label}"
    }
}
```
The `task_type` sets up the config for a specific task, entity_matching in this case.

Take a look at the prompt section of the config. This defines the settings related to defining the task and the machinery around it.  

The `task_guidelines` key is the most important key, it defines the task for the LLM to understand and execute on. In this case, we first set up the task and tell the model the kind of data present in the dataset, by telling it that it is an expert at identifying duplicate products. Next we explain the task to the model, saying that it has two identify if the given products are duplicate or not. We also make the output format clear by telling the model it has to choose from the options duplicate or not duplicate. 

The `example_template` is one of the most important keys to set for a task. This defines the format of every example that will be sent to the LLM. This creates a prompt using the columns from the input dataset, and sends this prompt to the LLM hoping for the llm to generate the column defined under the `label_column`, which is answer in our case. For every input, the model will be given the example with all the columns from the datapoint filled in according to the specification in the `example_template`. The `label_column` will be empty, and the LLM will generate the label. The `example_template` will be used to format all seed examples. Here we give the model both the entities separated by newlines and ask if the entities are duplicate or not duplicate.

The `few_shot_examples` here is a list of json inputs which define handpicked examples to use as seed examples for the model. These labeled examples help the model understand the task better and how it supposed to answer a question. If there is a larger number of examples, we can specify a path to a csv instead of a list of examples.

`few_shot_num` defines the number of examples selected from the seed set and sent to the LLM. Experiment with this number based on the input token budget and performance degradation with longer inputs.

`few_shot_selection` is set to fixed in this case as we want to use all examples as seed examples. However, if we want to use a subset of examples as seed examples from a larger set, we can set the appropriate strategy like `semantic_similarity` here to get dynamic good seed examples.

### Alternate config with multiple columns

Let's consider the case in which there are multiple columns in the dataset which are combined to create an input for the model.

```json
config = {
    "task_name": "ProductCatalogEntityMatch",
    "task_type": "entity_matching",
    "dataset": {
        "label_column": "label",
        "delimiter": ","
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo"
    },
    "prompt": {
        "task_guidelines": "You are an expert at identifying duplicate products from online product catalogs.\nYou will be given information about two product entities, and your job is to tell if they are the same (duplicate) or different (not duplicate). Your answer must be from one of the following options:\n{labels}",
        "labels": [
            "duplicate",
            "not duplicate"
        ],
        "example_template": "Title of entity1: {Title_entity1}; category of entity1: {Category_entity1}; brand of entity1: {Brand_entity1}; model number of entity1: {ModelNo_entity1}; price of entity1: {Price_entity1}\nTitle of entity2: {Title_entity2}; category of entity2: {Category_entity2}; brand of entity2: {Brand_entity2}; model number of entity2: {ModelNo_entity2}; price of entity2: {Price_entity2}\nDuplicate or not: {label}",
        "few_shot_examples": [
            {
                "Title_entity1": "lexmark extra high yield return pgm print cartridge - magenta",
                "Category_entity1": "printers",
                "Brand_entity1": "lexmark",
                "ModelNo_entity1": "c782u1mg",
                "Price_entity1": "214.88",
                "Title_entity2": "lexmark 18c1428 return program print cartridge black",
                "Category_entity2": "inkjet printer ink",
                "Brand_entity2": "lexmark",
                "ModelNo_entity2": "18c1428",
                "Price_entity2": "19.97",
                "label": "not duplicate"
            },
            {
                "Title_entity1": "edge tech proshot 4gb sdhc class 6 memory card",
                "Category_entity1": "usb drives",
                "Brand_entity1": "edge tech",
                "ModelNo_entity1": "pe209780",
                "Price_entity1": "10.88",
                "Title_entity2": "4gb edge proshot sdhc memory card class6",
                "Category_entity2": "computers accessories",
                "Brand_entity2": "edge",
                "ModelNo_entity2": "nan",
                "Price_entity2": "17.83",
                "label": "duplicate"
            }
        ],
        "few_shot_selection": "fixed",
        "few_shot_num": 2
    }
}
```

Notice how in this case, we specify how the different columns defining different aspects of every column are stitched together to form the final example template.

### Run the task

```py
from autolabel import LabelingAgent
agent = LabelingAgent(config)
agent.plan('data/walmart_amazon_test.csv')
agent.run('data/walmart_amazon_test.csv', max_items = 100)
```

### Evaluation metrics

On running the above config, this is an example output expected for labeling 100 items.
```
┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ support ┃ threshold ┃ accuracy ┃ completion_rate ┃
┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 100     │ -inf      │ 0.96     │ 1.0             │
└─────────┴───────────┴──────────┴─────────────────┘
```

**Accuracy** - This measures the proportion of examples which are marked correctly by the model - for eg which mark duplicate entities correctly.

**Completion Rate** - There can be errors while running the LLM related to labeling for eg. the LLM may give a label which is not in the label list or provide an answer which is not parsable by the library. In this cases, we mark the example as not labeled successfully. The completion rate refers to the proportion of examples that were labeled successfully.
