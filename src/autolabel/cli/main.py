from typing import Optional
from typing_extensions import Annotated
import logging

import typer

from autolabel import LabelingAgent
from autolabel.schema import TaskType, ModelProvider, FewShotAlgorithm
from autolabel.few_shot import PROVIDER_TO_MODEL
from autolabel.dataset import AutolabelDataset

from autolabel.cli.config import init, create_config_wizard

app = typer.Typer(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="[bold]Autolabel CLI[/bold] 🏷️",
)


@app.command(name="config")
def config_command(
    seed: Annotated[
        Optional[str],
        typer.Argument(
            help="Optional seed dataset to help auto-fill the config. Recommended for a more accurate config"
        ),
    ] = None,
):
    """Create a new [bold]config[/bold] file using a wizard 🪄"""
    create_config_wizard(seed)


@app.command(
    name="init",
)
def init_command(
    seed: Annotated[
        Optional[str],
        typer.Argument(
            help="Optional seed dataset to help auto-fill the config. Recommended for a more accurate config"
        ),
    ] = None,
    task_name: Annotated[
        str,
        typer.Option(
            help="Name of the task to create a config for",
            show_default=False,
        ),
    ] = None,
    task_type: Annotated[
        str,
        typer.Option(
            help=f"Type of task to create. Options: [magenta]{', '.join([t for t in TaskType])}[/magenta]",
            show_default=False,
        ),
    ] = None,
    dataset_label_column: Annotated[
        str,
        typer.Option(
            "--label-column",
            help="Name of the column containing the labels",
            show_default=False,
            rich_help_panel="Dataset Configuration",
        ),
    ] = None,
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
            help=f"Provider of the model to use. Options: [magenta]{', '.join([p for p in ModelProvider])}[/magenta]",
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
            help=f"Provider of the embedding model to use. Options: [magenta]{', '.join([p for p in PROVIDER_TO_MODEL])}[/magenta]",
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
    guess_labels: Annotated[
        bool,
        typer.Option(
            "--guess-labels",
            help="Whether to guess the labels from the seed dataset. If set, --task-type, --delimiter, and --label-column (and --label-separator for mulitlabel classification) must be defined",
            rich_help_panel="Prompt Configuration",
        ),
    ] = False,
    prompt_task_guidelines: Annotated[
        str,
        typer.Option(
            "--task-guidelines",
            help="Guidelines for the task. [code]{labels}[/code] will be replaced with a newline-separated list of labels",
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
    prompt_few_shot_selection: Annotated[
        str,
        typer.Option(
            "--few-shot-selection",
            help=f"What algorithm to use to select examples from the seed dataset. Options: [magenta]{', '.join([a for a in FewShotAlgorithm])}[/magenta]",
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    prompt_few_shot_num: Annotated[
        int,
        typer.Option(
            "--few-shot-num",
            help="Number of examples to select from the seed dataset",
            show_default=False,
            rich_help_panel="Prompt Configuration",
        ),
    ] = None,
    prompt_example_template: Annotated[
        str,
        typer.Option(
            "--example-template",
            help="Template to use for each example. [code]{column_name}[/code] will be replaced with the corresponding column value for each example",
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
):
    """Generate a new template [bold]config[/bold] file 📄"""
    init(
        seed,
        task_name,
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
        guess_labels=guess_labels,
        prompt_task_guidelines=prompt_task_guidelines,
        prompt_few_shot_examples=prompt_few_shot_examples,
        prompt_few_shot_selection=prompt_few_shot_selection,
        prompt_few_shot_num=prompt_few_shot_num,
        prompt_example_template=prompt_example_template,
        prompt_output_guidelines=prompt_output_guidelines,
        prompt_output_format=prompt_output_format,
        prompt_chain_of_thought=prompt_chain_of_thought,
    )


def setup_logging(
    verbose_debug: bool = False,
    verbose_info: bool = False,
    quiet_warning: bool = False,
    quiet_error: bool = False,
):
    if verbose_debug:
        log_level = logging.DEBUG
    elif verbose_info:
        log_level = logging.INFO
    elif quiet_warning:
        log_level = logging.ERROR
    elif quiet_error:
        log_level = logging.CRITICAL
    else:
        log_level = logging.WARNING
    logging.getLogger("autolabel").setLevel(log_level)
    logging.getLogger("langchain").setLevel(log_level)


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
    verbose_debug: Annotated[
        bool, typer.Option("--debug", "-vv", help="Verbose (debug log level)")
    ] = False,
    verbose_info: Annotated[
        bool, typer.Option("--info", "-v", help="Verbose (info log level)")
    ] = False,
    quiet_warning: Annotated[
        bool, typer.Option("--error", "-q", help="Quiet (error log level)")
    ] = False,
    quiet_error: Annotated[
        bool, typer.Option("--critical", "-qq", help="Quiet (critical log level)")
    ] = False,
):
    """[bold]Plan[/bold] 🔍 a labeling session in accordance with the provided dataset and config file"""
    setup_logging(verbose_debug, verbose_info, quiet_warning, quiet_error)
    agent = LabelingAgent(config=config, cache=cache)
    config = agent.config
    dataset = AutolabelDataset(dataset, config)
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
    output_name: Annotated[
        str, typer.Option(help="Path to output file to save run results to")
    ] = None,
    verbose_debug: Annotated[
        bool, typer.Option("--debug", "-vv", help="Verbose (debug log level)")
    ] = False,
    verbose_info: Annotated[
        bool, typer.Option("--info", "-v", help="Verbose (info log level)")
    ] = False,
    quiet_warning: Annotated[
        bool, typer.Option("--error", "-q", help="Quiet (error log level)")
    ] = False,
    quiet_error: Annotated[
        bool, typer.Option("--critical", "-qq", help="Quiet (critical log level)")
    ] = False,
):
    """[bold]Run[/bold] ▶️ a labeling session in accordance with the provided dataset and config file"""
    setup_logging(verbose_debug, verbose_info, quiet_warning, quiet_error)
    agent = LabelingAgent(config=config, cache=cache)
    config = agent.config
    dataset = AutolabelDataset(dataset, config)
    agent.run(
        dataset, max_items=max_items, output_name=output_name, start_index=start_index
    )


if __name__ == "__main__":
    app()
