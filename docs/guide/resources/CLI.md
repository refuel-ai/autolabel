The Autolabel CLI was created to make the config file creation process easier. It is a simple command line interface that will ask you a series of questions and then generate a config file for you. To use it, simply run the following command:

```bash
autolabel config
```

### Providing a seed file
You can provide a seed file to the CLI to help it generate the config file. Providing a seed file to the CLI allows it to automatically provide drop-down menus for column name inputs, detect labels that are already present in the seed file, and fill the few shot examples by row number in the seed file. To do this, simply run the following command:

```bash
autolabel config <path-to-seed-file>
```

For example, if you have a file called `seed.csv` in the current directory, you would run the following command:

```bash
autolabel config seed.csv
```

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
