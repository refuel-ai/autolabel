from typing import Optional, Dict
import csv
import json

import typer
from rich import print
from rich.prompt import Prompt, IntPrompt, Confirm

import pandas as pd

from autolabel.configs import AutolabelConfig, UnvalidatedAutolabelConfig
from autolabel.schema import TaskType, FewShotAlgorithm, ModelProvider
from autolabel.data_loaders import DatasetLoader


DEFAULT_TEXT_COLUMN = "example"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_EXAMPLE_TEMPLATE = "Example: {example}\nLabel: {label}"


def _create_dataset_config(task_type: TaskType, seed: Optional[str] = None) -> Dict:
    dataset_config = {}

    detected_delimiter = ","
    if seed:
        if seed.endswith(".csv"):
            with open(seed, "r") as f:
                dialect = csv.Sniffer().sniff(f.read(1024))
                f.seek(0)
            detected_delimiter = dialect.delimiter

    delimiter = Prompt.ask(
        "Enter the delimiter",
        default=detected_delimiter,
    )
    dataset_config[AutolabelConfig.DELIMITER_KEY] = delimiter

    text_column = Prompt.ask(
        "Enter the text column name",
        default=DEFAULT_TEXT_COLUMN,
    )
    dataset_config[AutolabelConfig.TEXT_COLUMN_KEY] = text_column

    label_column = Prompt.ask(
        "Enter the label column name", default=DEFAULT_LABEL_COLUMN
    )
    dataset_config[AutolabelConfig.LABEL_COLUMN_KEY] = label_column

    explanation_column = Prompt.ask(
        "Enter the explanation column name (optional)", default=None
    )
    if explanation_column:
        dataset_config[AutolabelConfig.EXPLANATION_COLUMN_KEY] = explanation_column

    if task_type == TaskType.MULTILABEL_CLASSIFICATION:
        label_separator = Prompt.ask(
            "Enter the label separator",
            default=";",
        )
        dataset_config[AutolabelConfig.LABEL_SEPARATOR_KEY] = label_separator

    return dataset_config


def _create_prompt_config(config: Dict, seed: Optional[str] = None) -> Dict:
    prompt_config = {}

    if seed:
        unvalidated_config = UnvalidatedAutolabelConfig(config)
        dataset_loader = DatasetLoader(seed, unvalidated_config, validate=False)

    task_guidelines = Prompt.ask(
        "Enter the task guidelines (optional)",
        default=None,
    )
    if task_guidelines:
        prompt_config[AutolabelConfig.TASK_GUIDELINE_KEY] = task_guidelines

    seed_labels = (
        dataset_loader.dat[unvalidated_config.label_column()].unique().tolist()
        if seed
        else []
    )
    if seed_labels and Confirm.ask(
        f"Detected labels in seed dataset.\n{seed_labels}\nUse these labels?"
    ):
        prompt_config[AutolabelConfig.VALID_LABELS_KEY] = seed_labels
    else:
        labels = []
        label = Prompt.ask("Enter a label (or leave blank to finish)")
        while label:
            labels.append(label)
            label = Prompt.ask("Enter a label (or leave blank to finish)")
        if labels:
            prompt_config[AutolabelConfig.VALID_LABELS_KEY] = labels

    if seed and Confirm.ask(f"Use {seed} as few shot example dataset?"):
        prompt_config[AutolabelConfig.FEW_SHOT_EXAMPLE_SET_KEY] = seed
    else:
        few_shot_example_set = []
        example = Prompt.ask("Enter an example or row number (or leave blank for none)")
        while example:
            if example.isdigit():
                example = dataset_loader.dat.iloc[int(example)][
                    AutolabelConfig.TEXT_COLUMN_KEY
                ]
                label = dataset_loader.dat.iloc[int(example)][
                    AutolabelConfig.LABEL_COLUMN_KEY
                ]
                print(example, label)
            else:
                label = Prompt.ask("Enter the label for this example", choices=labels)
            few_shot_example_set.append(
                {
                    config[AutolabelConfig.TEXT_COLUMN_KEY]: example,
                    config[AutolabelConfig.LABEL_COLUMN_KEY]: label,
                }
            )
            example = Prompt.ask(
                "Enter an example or row number (or leave blank to finish)"
            )
        if few_shot_example_set:
            prompt_config[
                AutolabelConfig.FEW_SHOT_EXAMPLE_SET_KEY
            ] = few_shot_example_set

    if AutolabelConfig.FEW_SHOT_EXAMPLE_SET_KEY in prompt_config:
        prompt_config[AutolabelConfig.FEW_SHOT_SELECTION_ALGORITHM_KEY] = Prompt.ask(
            "Enter the few shot selection algorithm",
            choices=[a for a in FewShotAlgorithm],
        )
        prompt_config[AutolabelConfig.FEW_SHOT_NUM_KEY] = IntPrompt.ask(
            "Enter the number of few shot examples to use", default=5
        )

    prompt_config[AutolabelConfig.EXAMPLE_TEMPLATE_KEY] = Prompt.ask(
        "Enter the example template",
        default=DEFAULT_EXAMPLE_TEMPLATE,
    ).replace("\\n", "\n")

    output_guideline = Prompt.ask(
        "Enter the output guideline (optional)",
        default=None,
    )
    if output_guideline:
        prompt_config[AutolabelConfig.OUTPUT_GUIDELINE_KEY] = output_guideline

    output_format = Prompt.ask(
        "Enter the output format (optional)",
        default=None,
    )
    if output_format:
        prompt_config[AutolabelConfig.OUTPUT_FORMAT_KEY] = output_format

    prompt_config[AutolabelConfig.CHAIN_OF_THOUGHT_KEY] = Confirm.ask(
        "Should the prompt use a chain of thought?", default=False
    )

    return prompt_config


def _create_model_config() -> Dict:
    model_config = {}

    model_config[AutolabelConfig.PROVIDER_KEY] = Prompt.ask(
        "Enter the model provider",
        choices=[p for p in ModelProvider],
    )

    model_config[AutolabelConfig.MODEL_NAME_KEY] = Prompt.ask("Enter the model name")

    model_params = {}
    model_param = Prompt.ask(
        "Enter a model parameter name (or leave blank for none)",
        default=None,
    )
    while model_param:
        model_param_value = Prompt.ask(
            f"Enter the value for {model_param}",
        )
        model_params[model_param] = model_param_value
        model_param = Prompt.ask(
            "Enter a model parameter name (or leave blank to finish)",
            default=None,
        )

    if model_params:
        model_config[AutolabelConfig.MODEL_PARAMS_KEY] = model_params

    model_config[AutolabelConfig.COMPUTE_CONFIDENCE_KEY] = Confirm.ask(
        "Should the model compute confidence?", default=False
    )

    model_config[AutolabelConfig.LOGIT_BIAS_KEY] = Confirm.ask(
        "Should the model use logit bias?", default=False
    )

    return model_config


def create_config(seed: Optional[str] = None):
    """Create a new task [bold]config[/bold] file"""
    config = {}
    config[AutolabelConfig.TASK_NAME_KEY] = Prompt.ask("Enter the task name")
    config[AutolabelConfig.TASK_TYPE_KEY] = Prompt.ask(
        "Enter the task type", choices=[t for t in TaskType]
    )

    config[AutolabelConfig.DATASET_CONFIG_KEY] = _create_dataset_config(
        config[AutolabelConfig.TASK_TYPE_KEY], seed
    )
    config[AutolabelConfig.PROMPT_CONFIG_KEY] = _create_prompt_config(config, seed)
    config[AutolabelConfig.MODEL_CONFIG_KEY] = _create_model_config()

    print(config)
    try:
        AutolabelConfig(config)
    except Exception as e:
        print(f"error validating config: {e}")
        if Confirm.ask("Would you like to fix the config?"):
            config = create_config(seed)
    print(f"Writing config to {config[AutolabelConfig.TASK_NAME_KEY]}_config.json")
    with open(
        f"{config[AutolabelConfig.TASK_NAME_KEY]}_config.json", "w"
    ) as config_file:
        json.dump(config, config_file, indent=4)
