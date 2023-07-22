from typing import Optional
from typing_extensions import Annotated

import typer

from autolabel import LabelingAgent
from autolabel.schema import TaskType, ModelProvider, FewShotAlgorithm
from autolabel.few_shot import PROVIDER_TO_MODEL

from autolabel.cli.config import create_config, create_config_wizard

app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="[bold]Autolabel CLI[/bold] 🏷️",
)


@app.command(
    name="config",
)
def create_config_command(
    task_name: Annotated[
        str,
        typer.Argument(
            help="Name of the task to create a config for", show_default=False
        ),
    ],
    seed: Annotated[
        Optional[str],
        typer.Argument(help="Optional seed dataset to help auto-fill the config."),
    ] = None,
    task_type: Annotated[
        str,
        typer.Option(
            "--type",
            help=f"Type of task to create. Options: {', '.join([t for t in TaskType])}",
            show_default=False,
        ),
    ] = None,
    dataset_label_column: Annotated[
        str,
        typer.Option(
            "--label-column",
            help="Name of the column containing the labels",
            rich_help_panel="Dataset Configuration",
        ),
    ] = "label",
    dataset_label_separator: Annotated[
        str,
        typer.Option(
            "--label-separator",
            help="Separator to use when separating multiple labels for multilabel classification",
            show_default=False,
            rich_help_panel="Dataset Configuration",
        ),
    ] = None,
    dataset_explanation_column: Annotated[
        str,
        typer.Option(
            "--explanation-column",
            help="Name of the column containing the explanations",
            show_default=False,
            rich_help_panel="Dataset Configuration",
        ),
    ] = None,
    dataset_text_column: Annotated[
        str,
        typer.Option(
            "--text-column",
            help="Name of the column containing the text to label",
            show_default=False,
            rich_help_panel="Dataset Configuration",
        ),
    ] = None,
    dataset_delimiter: Annotated[
        str,
        typer.Option(
            "--delimiter",
            help="Delimiter to use when parsing the dataset",
            show_default=False,
            rich_help_panel="Dataset Configuration",
        ),
    ] = None,  # None means it will be guessed from seed.csv or default to a comma
    model_provider: Annotated[
        str,
        typer.Option(
            "--provider",
            help=f"Provider of the model to use. Options: {', '.join([p for p in ModelProvider])}",
            rich_help_panel="Model Configuration",
        ),
    ] = "openai",
    model_name: Annotated[
        str,
        typer.Option(
            "--model",
            help="Name of the model to use",
            rich_help_panel="Model Configuration",
        ),
    ] = "gpt-3.5-turbo",
    model_compute_confidence: Annotated[
        bool,
        typer.Option(
            "--compute-confidence",
            help="Whether to compute confidence scores for each label",
            show_default=False,
            rich_help_panel="Model Configuration",
        ),
    ] = None,
    model_logit_bias: Annotated[
        bool,
        typer.Option(
            "--logit-bias",
            help="Whether to use logit biasing to constrain the model to certain tokens",
            show_default=False,
            rich_help_panel="Model Configuration",
        ),
    ] = None,
    embedding_provider: Annotated[
        str,
        typer.Option(
            "--embedding-provider",
            help=f"Provider of the embedding model to use. Options: {', '.join([p for p in PROVIDER_TO_MODEL])}",
            show_default=False,
            rich_help_panel="Embedding Configuration",
        ),
    ] = None,
    embedding_model_name: Annotated[
        str,
        typer.Option(
            "--embedding-model",
            help="Name of the embedding model to use",
            show_default=False,
            rich_help_panel="Embedding Configuration",
        ),
    ] = None,
    prompt_task_guidelines: Annotated[
        str,
        typer.Option(
            "--task-guidelines",
            help='Guidelines for the task. "{labels}" will be replaced with a newline-separated list of labels',
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    prompt_few_shot_examples: Annotated[
        str,
        typer.Option(
            "--few-shot-examples",
            help="Seed dataset to use for few-shot prompting",
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    prompt_example_selection: Annotated[
        str,
        typer.Option(
            "--example-selection",
            help=f"What algorithm to use to select examples from the seed dataset. Options: {', '.join([a for a in FewShotAlgorithm])}",
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    prompt_num_examples: Annotated[
        int,
        typer.Option(
            "--num-examples",
            help="Number of examples to select from the seed dataset",
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    prompt_example_template: Annotated[
        str,
        typer.Option(
            "--example-template",
            help='Template to use for each example. "{column_name}" will be replaced with the corresponding column value for each example',
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    prompt_output_guidelines: Annotated[
        str,
        typer.Option(
            "--output-guidelines",
            help="Guidelines for the output",
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    prompt_output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            help="Format to use for the output",
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    prompt_chain_of_thought: Annotated[
        bool,
        typer.Option(
            "--chain-of-thought",
            help="Whether to use chain of thought",
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    wizard: Annotated[
        bool,
        typer.Option(help="Use step-by-step wizard to create config", is_flag=True),
    ] = False,
):
    """Create a new task [bold]config[/bold] file


    To use column values in the example template, surround the column name with curly braces.
    """
    if wizard:
        create_config_wizard(task_name, seed, task_type=task_type)
    else:
        create_config(
            task_name,
            seed,
            task_type,
            dataset_label_column=dataset_label_column,
            dataset_label_separator=dataset_label_separator,
            dataset_explanation_column=dataset_explanation_column,
            dataset_text_column=dataset_text_column,
            dataset_delimiter=dataset_delimiter,
            model_provider=model_provider,
            model_name=model_name,
            model_compute_confidence=model_compute_confidence,
            model_logit_bias=model_logit_bias,
            embedding_provider=embedding_provider,
            embedding_model_name=embedding_model_name,
            prompt_task_guidelines=prompt_task_guidelines,
            prompt_few_shot_examples=prompt_few_shot_examples,
            prompt_example_selection=prompt_example_selection,
            prompt_num_examples=prompt_num_examples,
            prompt_example_template=prompt_example_template,
            prompt_output_guidelines=prompt_output_guidelines,
            prompt_output_format=prompt_output_format,
            prompt_chain_of_thought=prompt_chain_of_thought,
        )


@app.command()
def plan(
    dataset: Annotated[
        str, typer.Argument(help="Path to dataset to label", show_default=False)
    ],
    config: Annotated[
        str, typer.Argument(help="Path to config file", show_default=False)
    ],
    max_items: Annotated[int, typer.Option(help="Max number of items to label")] = None,
    start_index: Annotated[int, typer.Option(help="Index to start at")] = 0,
    cache: Annotated[bool, typer.Option(help="Cache results")] = True,
):
    """[bold]Plan[/bold] a labeling session in accordance with the provided dataset and config file"""
    agent = LabelingAgent(config=config, cache=cache)
    agent.plan(dataset, max_items=max_items, start_index=start_index)


@app.command()
def run(
    dataset: Annotated[
        str, typer.Argument(help="Path to dataset to label", show_default=False)
    ],
    config: Annotated[
        str, typer.Argument(help="Path to config file", show_default=False)
    ],
    max_items: Annotated[int, typer.Option(help="Max number of items to label")] = None,
    start_index: Annotated[int, typer.Option(help="Index to start at")] = 0,
    cache: Annotated[bool, typer.Option(help="Cache results")] = True,
):
    """[bold]Run[/bold] a labeling session in accordance with the provided dataset and config file"""
    agent = LabelingAgent(config=config, cache=cache)
    agent.run(dataset, max_items=max_items, start_index=start_index)


if __name__ == "__main__":
    app()
