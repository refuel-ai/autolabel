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
    task_type: Annotated[
        str,
        typer.Argument(
            help=f"Type of task to create a config for. Options: {', '.join([t for t in TaskType])}",
            show_default=False,
        ),
    ],
    seed: Annotated[
        Optional[str],
        typer.Argument(help="Optional seed dataset to help auto-fill the config."),
    ] = None,
    label_column: Annotated[
        str,
        typer.Option(
            help="Name of the column containing the labels",
            rich_help_panel="Dataset Configuration",
        ),
    ] = "label",
    label_separator: Annotated[
        str,
        typer.Option(
            help="Separator to use when separating multiple labels for multilabel classification",
            show_default=False,
            rich_help_panel="Dataset Configuration",
        ),
    ] = ";",
    explanation_column: Annotated[
        str,
        typer.Option(
            help="Name of the column containing the explanations",
            show_default=False,
            rich_help_panel="Dataset Configuration",
        ),
    ] = None,
    text_column: Annotated[
        str,
        typer.Option(
            help="Name of the column containing the text to label",
            rich_help_panel="Dataset Configuration",
        ),
    ] = "example",
    delimiter: Annotated[
        str,
        typer.Option(
            help="Delimiter to use when parsing the dataset",
            show_default=False,
            rich_help_panel="Dataset Configuration",
        ),
    ] = None,  # None means it will be guessed from seed.csv or default to a comma
    provider: Annotated[
        str,
        typer.Option(
            help=f"Provider of the model to use. Options: {', '.join([p for p in ModelProvider])}",
            rich_help_panel="Model Configuration",
        ),
    ] = "openai",
    model_name: Annotated[
        str,
        typer.Option(
            help="Name of the model to use",
            rich_help_panel="Model Configuration",
        ),
    ] = "gpt-3.5-turbo",
    compute_confidence: Annotated[
        bool,
        typer.Option(
            help="Whether to compute confidence scores for each label",
            rich_help_panel="Model Configuration",
        ),
    ] = False,
    logit_bias: Annotated[
        bool,
        typer.Option(
            help="Whether to use logit biasing to constrain the model to certain tokens",
            rich_help_panel="Model Configuration",
        ),
    ] = False,
    embedding_provider: Annotated[
        str,
        typer.Option(
            help=f"Provider of the embedding model to use. Options: {', '.join([p for p in PROVIDER_TO_MODEL])}",
            rich_help_panel="Embedding Configuration",
        ),
    ] = "openai",
    embedding_model_name: Annotated[
        str,
        typer.Option(
            help="Name of the embedding model to use",
            show_default=False,
            rich_help_panel="Embedding Configuration",
        ),
    ] = None,
    task_guidelines: Annotated[
        str,
        typer.Option(
            help="Guidelines for the task",
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    use_seed: Annotated[
        bool,
        typer.Option(
            help="Whether to use the seed dataset as the example set",
            rich_help_panel="Prompt Configuration",
        ),
    ] = True,
    example_selection: Annotated[
        str,
        typer.Option(
            help=f"What algorithm to use to select examples from the seed dataset. Options: {', '.join([a for a in FewShotAlgorithm])}",
            rich_help_panel="Prompt Configuration",
        ),
    ] = "fixed",
    num_examples: Annotated[
        int,
        typer.Option(
            help="Number of examples to select from the seed dataset",
            rich_help_panel="Prompt Configuration",
        ),
    ] = 5,
    example_template: Annotated[
        str,
        typer.Option(
            help="Template to use for each example",
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    output_guidelines: Annotated[
        str,
        typer.Option(
            help="Guidelines for the output",
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            help="Format to use for the output",
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    chain_of_thought: Annotated[
        bool,
        typer.Option(
            help="Whether to use chain of thought",
            rich_help_panel="Prompt Configuration",
        ),
    ] = False,
    wizard: Annotated[
        bool,
        typer.Option(help="Use step-by-step wizard to create config", is_flag=True),
    ] = False,
):
    """Create a new task [bold]config[/bold] file

    To use the labels list in the task guidelines, surround the word "labels" with curly braces.
    To use column values in the example template, surround the column name with curly braces.
    """
    if wizard:
        create_config_wizard(task_name, task_type, seed)
    else:
        create_config(task_name, task_type, seed)


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
