from typing import Optional
from typing_extensions import Annotated

import typer

from autolabel import LabelingAgent

from autolabel.cli.config import create_config

app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="[bold]Autolabel CLI[/bold] 🏷️",
)


@app.command(
    name="create-config",
)
def create_config_command(
    seed: Annotated[
        Optional[str],
        typer.Argument(help="Optional seed dataset to help auto-fill the config."),
    ] = None
):
    """Create a new task [bold]config[/bold] file

    To use the labels list in the task guidelines, surround the word "labels" with curly braces.
    To use column values in the example template, surround the column name with curly braces.
    """
    create_config(seed)


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
