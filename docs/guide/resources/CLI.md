The Autolabel CLI was created to make the [config](https://docs.refuel.ai/guide/resources/configs) file creation process easier. It is a simple command line interface that will ask you a series of questions and then generate a config file for you. To use it, simply run the following command:

```bash
autolabel config
```

**Walkthrough: Creating a Config for Civil Comments**

1. The first step is to run the `autolabel` command with the `config` argument:

```bash
autolabel config
```

2. The program will prompt you to enter the task name.

```
Enter the task name: ToxicCommentClassification
```

3. Next, you need to choose the task type from the provided options.:

```
Choose a task type
> classification
  named_entity_recognition
  question_answering
  entity_matching
  multilabel_classification
```

4. Now, the program will ask for dataset configuration details. You need to specify the delimiter used in your dataset, the label column name, and an optional explanation column name:

```
Dataset Configuration
Enter the delimiter (,):
Enter the label column name: label
Enter the explanation column name (optional):
```

*Anything surrounded by parenthesis at the end of a prompt will be used as the default value if you don't input anything. Make sure to change this if it does not line up with your task.*

5. The program will then ask for model configuration. You will need to specify the model provider from the options. Next, enter the model name, optional model parameters, whether the model should compute confidence, and the strength of the logit bias:

```
Model Configuration
Enter the model provider
> openai
  anthropic
  huggingface_pipeline
  refuel
  google
  cohere
Enter the model name: gpt-3.5-turbo
Enter a model parameter name (or leave blank for none):
Should the model compute confidence? [y/n] (n):
What is the strength of logit bias? (0.0): 100
```

6. Next, you will configure the task prompt. First, enter the task guidelines. In the task guidelines, `{num_labels}` and `{labels}` will be replaced by the number of labels and the labels list respectively. Next, specify the labels. Then, write the example template with placeholders for the column names you want to use in the prompt. You can also add an output guideline and format if needed. Lastly, you can choose whether to use a chain of thought:

```
Prompt Configuration
Enter the task guidelines (Your job is to correctly label the provided input example into one of the following {num_labels} categories.
Categories:
{labels}
):
Enter a valid label (or leave blank for none): toxic
Enter a valid label (or leave blank to finish): not toxic
Enter a valid label (or leave blank to finish):
Enter the example template: Example: {example}\nLabel: {label}
Enter the value for example (or leave blank for none):
Enter the output guideline (optional):
Enter the output format (optional):
Should the prompt use a chain of thought? [y/n] (n):
```

7. The program will then display the configuration that you have provided as a python dictionary:

```python
{
    'task_name': 'ToxicCommentClassification',
    'task_type': 'classification',
    'dataset': {'delimiter': ',', 'label_column': 'label'},
    'model': {'provider': 'openai', 'name': 'gpt-3.5-turbo', 'compute_confidence': False, 'logit_bias': 100.0},
    'prompt': {
        'task_guidelines': 'Your job is to correctly label the provided input example into one of the following {num_labels} categories.\nCategories:\n{labels}\n',
        'labels': ['toxic', 'not toxic'],
        'example_template': 'Example: {example}\nLabel: {label}',
        'chain_of_thought': False
    }
}
```

8. Finally, the program will write the configuration to a file named "{your_task_name}_config.json".

```
Writing config to ToxicCommentClassification_config.json
```

That's it! You have successfully created a config for a task using the CLI program. The generated configuration file can now be used for any labeling runs with autolabel!

### Providing a seed file
You can provide a seed file to the CLI to help it generate the config file. Providing a seed file to the CLI allows it to automatically provide drop-down menus for column name inputs, detect labels that are already present in the seed file, and fill the few shot examples by row number in the seed file. To do this, simply run the following command:

```bash
autolabel config <path-to-seed-file>
```

For example, if you have a file called `seed.csv` in the current directory, you would run the following command:

```bash
autolabel config seed.csv
```

Here's an example of what the prompt configuration section would look like with a seed file:

```
Detected 2 unique labels in seed dataset. Use these labels? [y/n]: y
Enter the example template: Example: {example}\nLabel: {label}
Use seed.csv as few shot example dataset? [y/n]: n
Enter the value for example or row number (or leave blank for none): 3
{'example': "When all else fails, change the subject to Hillary's emails.", 'label': 'not toxic'}
Enter the value for example or row number (or leave blank to finish): 7
{
    'example': 'He may like the internal forum, but the reality is he has affirmed traditional doctrine and practices. While he does like the internal forum he has not changed anything.',
    'label': 'not toxic'
}
Enter the value for example or row number (or leave blank to finish): 24
{'example': '........... said the blind dumb and deaf lemming.', 'label': 'toxic'}
Enter the value for example or row number (or leave blank to finish): 64
{
    'example': 'Do you have a citation for that statement or did you just make it up yourself? BTW, this thread is about the unhealthy liar the Democrats have
nominated.',
    'label': 'not toxic'
}
Enter the value for example or row number (or leave blank to finish):
Enter the few shot selection algorithm
> fixed
  semantic_similarity
  max_marginal_relevance
  label_diversity_random
  label_diversity_similarity
Enter the number of few shot examples to use (4):
```

As you can see, the CLI automatically detected the labels in the seed file and used them to generate the labels list. It also automatically filled the few shot examples with the examples from the seed file after letting the user choose the rows to use.

### Specifying Model Parameters
To specify model parameters, you can simply enter the parameter name and value when prompted. For example, if you wanted to specify the `temperature` parameter for the `gpt-3.5-turbo` model, you would run the following command:

```
Enter a model parameter name (or leave blank for none): temperature
Enter the value for max_tokens: 0.5
```

### Providing Few Shot Examples
To provide few shot examples, you can simply input the example when prompted (after entering the example template). The CLI will go through the example template and ask for any values specified in that. For example, if you template is `Example: {example}\nLabel: {label}`, you could add a few shot example as shown below:

```
Enter the example template: Example: {example}\nLabel: {label}
Enter the value for example (or leave blank for none): You're ugly and dumb
Enter the value for label: toxic
Enter the value for example (or leave blank to finish): I love your art!
Enter the value for label: not toxic
Enter the value for example (or leave blank to finish): It was a great show. Not a combo I'd of expected to be good together but it was.
Enter the value for label: not toxic
Enter the value for example (or leave blank to finish): It's ridiculous that these guys are being called 'protesters'. Being armed is a threat of violence, which makes them terrorists
Enter the value for label: toxic
Enter the value for example (or leave blank to finish):
Enter the few shot selection algorithm
> fixed
  semantic_similarity
  max_marginal_relevance
  label_diversity_random
  label_diversity_similarity
Enter the number of few shot examples to use (4):
```

Since we only added 4 examples, we chose the `fixed` few shot selection algorithm and left the number of few shot examples to use at 4 since we want to use all of them in every prompt.

## The `init` command
If you would prefer to edit a json file directly, you can use the `init` command to generate a config file for you. To do this, simply run the following command:

```bash
autolabel init
```

By default, this will create a config file that looks like the one below: 

```json
{
    "task_name": "[TODO] Enter task name",
    "task_type": "[TODO] Enter task type",
    "dataset": {
        "delimiter": "[TODO] Enter delimiter",
        "label_column": "[TODO] Enter label column name"
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo",
        "params": {}
    },
    "prompt": {
        "task_guidelines": "[TODO] Enter task guidelines",
        "example_template": "[TODO] Enter example template",
        "few_shot_examples": "[TODO] Enter few shot examples",
        "few_shot_selection": "[TODO] Enter few shot selection",
        "few_shot_num": "[TODO] Enter few shot num"
    }
}
```

`init` will also take a seed file as an argument. Combined with other options, this can result in a very quick config file generation process. For example, if you have a file called `seed.csv` in the current directory, you could run the following command:

```bash
autolabel init seed.csv --task-name ToxicCommentClassification --task-type classification --delimiter , --label-column label --task-guidelines "You are an expert at identifying toxic comments." --example-template "Example: {example}\nLabel: {label}" --few-shot-examples seed.csv --few-shot-selection semantic_similarity --few-shot-num 5 --guess-labels
```

Resulting in the following config file for the civil comments dataset:

```json
{
    "task_name": "ToxicCommentClassification",
    "task_type": "classification",
    "dataset": {
        "delimiter": ",",
        "label_column": "label"
    },
    "model": {
        "provider": "openai",
        "name": "gpt-3.5-turbo",
        "params": {}
    },
    "prompt": {
        "task_guidelines": "You are an expert at identifying toxic comments.",
        "example_template": "Example: {example}\nLabel: {label}",
        "few_shot_examples": "seed.csv",
        "few_shot_selection": "semantic_similarity",
        "few_shot_num": 5,
        "labels": [
            "not toxic",
            "toxic"
        ]
    }
}
```

## The `plan` command
The `plan` command works identically to running `LabelingAgent({config}).plan({dataset})` in python. To use it, simply run the following command:

```bash
autolabel plan <path-to-dataset> <path-to-config>
```

## The `run` command
The `run` command works identically to running `LabelingAgent({config}).run({dataset})` in python. To use it, simply run the following command:

```bash
autolabel run <path-to-dataset> <path-to-config>
```

## Help
If any of the commands are unclear, you can run `autolabel --help` to see the help menu or `autolabel <command> --help` to see the help menu for a specific command.