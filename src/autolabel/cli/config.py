from typing import Optional, Dict, List
import csv
import json

import typer
from rich import print
from rich.prompt import Prompt, IntPrompt, Confirm
from simple_term_menu import TerminalMenu

import pandas as pd

from autolabel.configs import AutolabelConfig as ALC
from autolabel.configs.schema import schema
from autolabel.schema import TaskType, FewShotAlgorithm, ModelProvider
from autolabel.data_loaders import DatasetLoader
from autolabel.models import ModelFactory
from autolabel.tasks import TASK_TYPE_TO_IMPLEMENTATION

from autolabel.cli.template import (
    TEMPLATE_CONFIG,
    TEMPLATE_TASK_NAME,
    TEMPLATE_LABEL_SEPARATOR,
)


def _get_sub_config(key: str, **kwargs) -> Dict:
    config = {}
    for p in schema["properties"][key]["properties"]:
        if f"{key}_{p}" in kwargs and kwargs[f"{key}_{p}"] is not None:
            if isinstance(kwargs[f"{key}_{p}"], str):
                config[p] = kwargs[f"{key}_{p}"].replace("\\n", "\n")
            else:
                config[p] = kwargs[f"{key}_{p}"]
    return {**TEMPLATE_CONFIG[key], **config}


def _get_labels_from_seed(df: pd.DataFrame, config: Dict) -> List[str]:
    if config[ALC.TASK_TYPE_KEY] in [
        TaskType.CLASSIFICATION.value,
        TaskType.ENTITY_MATCHING.value,
    ]:
        return (
            df[config[ALC.DATASET_CONFIG_KEY][ALC.LABEL_COLUMN_KEY]].unique().tolist()
        )
    elif config[ALC.TASK_TYPE_KEY] == TaskType.NAMED_ENTITY_RECOGNITION.value:
        return list(
            pd.json_normalize(
                df[config[ALC.DATASET_CONFIG_KEY][ALC.LABEL_COLUMN_KEY]].apply(
                    json.loads
                )
            ).columns
        )
    elif config[ALC.TASK_TYPE_KEY] == TaskType.MULTILABEL_CLASSIFICATION.value:
        return (
            df[config[ALC.DATASET_CONFIG_KEY][ALC.LABEL_COLUMN_KEY]]
            .str.split(config[ALC.DATASET_CONFIG_KEY][ALC.LABEL_SEPARATOR_KEY])
            .explode()
            .unique()
            .tolist()
        )


def init(
    seed: Optional[str] = None,
    task_name: Optional[str] = None,
    task_type: Optional[str] = None,
    guess_labels: bool = False,
    **kwargs,
):
    if not task_name:
        task_name = TEMPLATE_CONFIG[ALC.TASK_NAME_KEY]
    try:
        TaskType(task_type)
    except ValueError:
        task_type = TEMPLATE_CONFIG[ALC.TASK_TYPE_KEY]
    config = {ALC.TASK_NAME_KEY: task_name, ALC.TASK_TYPE_KEY: task_type}
    if (
        task_type == TaskType.MULTILABEL_CLASSIFICATION.value
        and kwargs["dataset_label_separator"] is None
    ):
        kwargs["dataset_label_separator"] = TEMPLATE_LABEL_SEPARATOR
    config[ALC.DATASET_CONFIG_KEY] = _get_sub_config("dataset", **kwargs)
    config[ALC.MODEL_CONFIG_KEY] = _get_sub_config("model", **kwargs)

    if guess_labels and seed:
        try:
            df = pd.read_csv(
                seed,
                delimiter=config[ALC.DATASET_CONFIG_KEY][ALC.DELIMITER_KEY],
                nrows=100,
            )
            labels = _get_labels_from_seed(df, config)
            if labels:
                kwargs["prompt_labels"] = labels
        except Exception:
            print("[red]Failed to infer labels[/red]")

    # TODO: add automatic example template generation
    config[ALC.PROMPT_CONFIG_KEY] = _get_sub_config("prompt", **kwargs)

    print(config)
    config_name = (
        config[ALC.TASK_NAME_KEY]
        if config[ALC.TASK_NAME_KEY] != TEMPLATE_TASK_NAME
        else "template"
    )
    print(f"Writing config to {config_name}_config.json")
    with open(f"{config_name}_config.json", "w") as config_file:
        json.dump(config, config_file, indent=4)


def _create_dataset_config_wizard(
    task_type: TaskType, seed: Optional[str] = None
) -> Dict:
    print("[bold]Dataset Configuration[/bold]")
    dataset_config = {}

    detected_delimiter = ","
    if seed:
        if seed.endswith(".csv"):
            try:
                with open(seed, "r") as f:
                    dialect = csv.Sniffer().sniff(f.read(1024))
                    f.seek(0)
                detected_delimiter = dialect.delimiter
            except Exception:
                pass

    delimiter = Prompt.ask(
        "Enter the delimiter",
        default=detected_delimiter,
    )
    dataset_config[ALC.DELIMITER_KEY] = delimiter
    if seed:
        df = pd.read_csv(seed, delimiter=delimiter, nrows=0)
        column_names = df.columns.tolist()
        label_column = column_names[
            TerminalMenu(column_names, title="Choose a label column").show()
        ]
        dataset_config[ALC.LABEL_COLUMN_KEY] = label_column
        options = [None] + column_names
        explanation_column = options[
            TerminalMenu(options, title="Choose an explanation column").show()
        ]
        if explanation_column:
            dataset_config[ALC.EXPLANATION_COLUMN_KEY] = explanation_column
    else:
        label_column = Prompt.ask("Enter the label column name")
        while not label_column:
            print("[red]The label column name cannot be blank[/red]")
            label_column = Prompt.ask("Enter the label column name")
        dataset_config[ALC.LABEL_COLUMN_KEY] = label_column

        explanation_column = Prompt.ask(
            "Enter the explanation column name (optional)", default=None
        )
        if explanation_column:
            dataset_config[ALC.EXPLANATION_COLUMN_KEY] = explanation_column

    if task_type == TaskType.MULTILABEL_CLASSIFICATION:
        label_separator = Prompt.ask(
            "Enter the label separator",
            default=";",
        )
        dataset_config[ALC.LABEL_SEPARATOR_KEY] = label_separator

    return dataset_config


def _create_model_config_wizard() -> Dict:
    print("[bold]Model Configuration[/bold]")
    model_config = {}

    options = [p.value for p in ModelProvider]
    model_config[ALC.PROVIDER_KEY] = options[
        TerminalMenu(options, title="Enter the model provider").show()
    ]

    model_config[ALC.MODEL_NAME_KEY] = Prompt.ask("Enter the model name")

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
        model_config[ALC.MODEL_PARAMS_KEY] = model_params

    model_config[ALC.COMPUTE_CONFIDENCE_KEY] = Confirm.ask(
        "Should the model compute confidence?", default=False
    )

    model_config[ALC.LOGIT_BIAS_KEY] = Confirm.ask(
        "Should the model use logit bias?", default=False
    )

    return model_config


def _create_prompt_config_wizard(config: Dict, seed: Optional[str] = None) -> Dict:
    print("[bold]Prompt Configuration[/bold]")
    prompt_config = {}

    if seed:
        unvalidated_config = ALC(config, validate=False)
        dataset_loader = DatasetLoader(seed, unvalidated_config, validate=False)

    prompt_config[ALC.TASK_GUIDELINE_KEY] = Prompt.ask(
        "Enter the task guidelines",
        default=TASK_TYPE_TO_IMPLEMENTATION[
            TaskType(config[ALC.TASK_TYPE_KEY])
        ].DEFAULT_TASK_GUIDELINES,
    ).replace("\\n", "\n")

    seed_labels = (
        dataset_loader.dat[unvalidated_config.label_column()].unique().tolist()
        if seed
        else []
    )
    if seed_labels and Confirm.ask(
        f"Detected {len(seed_labels)} unique labels in seed dataset. Use these labels?"
    ):
        prompt_config[ALC.VALID_LABELS_KEY] = seed_labels
    else:
        labels = []
        label = Prompt.ask("Enter a valid label (or leave blank for none)")
        while label:
            labels.append(label)
            label = Prompt.ask("Enter a valid label (or leave blank to finish)")
        if labels:
            prompt_config[ALC.VALID_LABELS_KEY] = labels

    prompt_config[ALC.EXAMPLE_TEMPLATE_KEY] = Prompt.ask(
        "Enter the example template",
    ).replace("\\n", "\n")
    example_template_variables = [
        v.split("}")[0].split("{")[1]
        for v in prompt_config[ALC.EXAMPLE_TEMPLATE_KEY].split(" ")
        if "{" in v and "}" in v
    ]
    while (
        config[ALC.DATASET_CONFIG_KEY][ALC.LABEL_COLUMN_KEY]
        not in example_template_variables
    ):
        print(
            "[red]The label column name must be included in the example template[/red]"
        )
        prompt_config[ALC.EXAMPLE_TEMPLATE_KEY] = Prompt.ask(
            "Enter the example template",
        ).replace("\\n", "\n")
        example_template_variables = [
            v.split("}")[0].split("{")[1]
            for v in prompt_config[ALC.EXAMPLE_TEMPLATE_KEY].split(" ")
            if "{" in v and "}" in v
        ]

    if seed and Confirm.ask(f"Use {seed} as few shot example dataset?"):
        prompt_config[ALC.FEW_SHOT_EXAMPLE_SET_KEY] = seed
    else:
        few_shot_example_set = []
        example = Prompt.ask(
            f"Enter the value for {example_template_variables[0]} {'or row number ' if seed else ''}(or leave blank for none)"
        )
        while example:
            example_dict = {}
            if seed and example.isdigit():
                example_dict = dataset_loader.dat.iloc[int(example)].to_dict()
                print(example_dict)
            else:
                example_dict[example_template_variables[0]] = example
                for variable in example_template_variables[1:]:
                    example_dict[variable] = Prompt.ask(
                        f"Enter the value for {variable}"
                    )

            few_shot_example_set.append(example_dict)
            example = Prompt.ask(
                f"Enter the value for {example_template_variables[0]} {'or row number ' if seed else ''}(or leave blank to finish)"
            )
        if few_shot_example_set:
            prompt_config[ALC.FEW_SHOT_EXAMPLE_SET_KEY] = few_shot_example_set

    if ALC.FEW_SHOT_EXAMPLE_SET_KEY in prompt_config:
        options = [a.value for a in FewShotAlgorithm]
        prompt_config[ALC.FEW_SHOT_SELECTION_ALGORITHM_KEY] = options[
            TerminalMenu(options, title="Enter the few shot selection algorithm").show()
        ]
        prompt_config[ALC.FEW_SHOT_NUM_KEY] = IntPrompt.ask(
            "Enter the number of few shot examples to use",
            default=min(len(prompt_config[ALC.FEW_SHOT_EXAMPLE_SET_KEY]), 5),
        )

    output_guideline = Prompt.ask(
        "Enter the output guideline (optional)",
        default=None,
    )
    if output_guideline:
        prompt_config[ALC.OUTPUT_GUIDELINE_KEY] = output_guideline

    output_format = Prompt.ask(
        "Enter the output format (optional)",
        default=None,
    )
    if output_format:
        prompt_config[ALC.OUTPUT_FORMAT_KEY] = output_format

    prompt_config[ALC.CHAIN_OF_THOUGHT_KEY] = Confirm.ask(
        "Should the prompt use a chain of thought?", default=False
    )

    return prompt_config


def create_config_wizard(
    seed: Optional[str] = None,
    **kwargs,
):
    """Create a new task [bold]config[/bold] file"""
    config = {}
    task_name = Prompt.ask("Enter the task name")
    config[ALC.TASK_NAME_KEY] = task_name
    options = [t.value for t in TaskType]
    config[ALC.TASK_TYPE_KEY] = options[
        TerminalMenu(options, title="Choose a task type").show()
    ]

    config[ALC.DATASET_CONFIG_KEY] = _create_dataset_config_wizard(
        config[ALC.TASK_TYPE_KEY], seed
    )
    config[ALC.MODEL_CONFIG_KEY] = _create_model_config_wizard()
    config[ALC.PROMPT_CONFIG_KEY] = _create_prompt_config_wizard(config, seed)

    print(config)
    try:
        ALC(config)
    except Exception as e:
        print(f"error validating config: {e}")
        if Confirm.ask("Would you like to fix the config?"):
            config = create_config(seed)
    print(f"Writing config to {config[ALC.TASK_NAME_KEY]}_config.json")
    with open(f"{config[ALC.TASK_NAME_KEY]}_config.json", "w") as config_file:
        json.dump(config, config_file, indent=4)
