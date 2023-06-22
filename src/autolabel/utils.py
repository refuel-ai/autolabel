import hashlib
import os
import json
import logging
from string import Formatter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
import shutil

import regex
import wget
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    ProgressType,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

logger = logging.getLogger(__name__)

EXAMPLE_DATASETS = [
    "banking",
    "civil_comments",
    "ledgar",
    "walmart_amazon",
    "company",
    "squad_v2",
    "sciq",
    "conll2003",
    "movie_reviews",
]

NO_SEED_DATASET = [
    "movie_reviews",
]

DATASET_URL = "https://autolabel-benchmarking.s3.us-west-2.amazonaws.com/{dataset}/{partition}.csv"


def extract_valid_json_substring(string: str) -> str:
    pattern = (
        r"{(?:[^{}]|(?R))*}"  # Regular expression pattern to match a valid JSON object
    )
    match = regex.search(pattern, string)
    if match:
        json_string = match.group(0)
        try:
            json.loads(json_string)
            return json_string
        except ValueError:
            pass
    return None


def calculate_md5(input_data: Any) -> str:
    if isinstance(input_data, dict):
        # Convert dictionary to a JSON-formatted string
        input_str = json.dumps(input_data, sort_keys=True).encode("utf-8")
    elif hasattr(input_data, "read"):
        # Read binary data from file-like object
        md5_hash = hashlib.md5()
        for chunk in iter(lambda: input_data.read(4096), b""):
            md5_hash.update(chunk)
        return md5_hash.hexdigest()
    elif isinstance(input_data, list):
        md5_hash = hashlib.md5()
        for item in input_data:
            md5_hash.update(calculate_md5(item).encode("utf-8"))
        return md5_hash.hexdigest()
    else:
        # Convert other input to byte string
        input_str = str(input_data).encode("utf-8")

    # Calculate MD5 hash of byte string
    md5_hash = hashlib.md5(input_str)
    return md5_hash.hexdigest()


def get_format_variables(fmt_string: str) -> List:
    return [i[1] for i in Formatter().parse(fmt_string) if i[1] is not None]


def _autolabel_progress(
    description: str = None,
    console: Optional[Console] = None,
    transient: bool = False,
    disable: bool = False,
) -> Progress:
    """Create a progress bar for autolabel."""
    columns: List[ProgressColumn] = (
        [TextColumn("[progress.description]{task.description}")] if description else []
    )
    columns.extend(
        (
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
    )
    return Progress(
        *columns,
        console=console,
        transient=transient,
        disable=disable,
    )


def track(
    sequence: Union[Sequence[ProgressType], Iterable[ProgressType]],
    description: str = None,
    total: Optional[int] = None,
    advance: int = 1,
    transient: bool = False,
    console: Optional[Console] = None,
    disable: bool = False,
) -> Iterable[ProgressType]:
    """Track progress by iterating over a sequence.

    Args:
        sequence (Iterable[ProgressType]): A sequence (must support "len") you wish to iterate over.
        description (str, optional): Description of task show next to progress bar. Defaults to `None`.
        total (int, optional): Total number of steps. Default is len(sequence).
        advance (int, optional): Number of steps to advance progress by. Defaults to 1. Total / advance must less than or equal to len(sequence) for progress to reach finished state.
        transient (bool, optional): Clear the progress on exit. Defaults to False.
        console (Console, optional): Console to write to. Default creates internal Console instance.
        disable (bool, optional): Disable display of progress.
    Returns:
        Iterable[ProgressType]: An iterable of the values in the sequence.
    """
    progress = _autolabel_progress(
        description=description,
        transient=transient,
        console=console,
        disable=disable,
    )

    if total is None:
        total = len(sequence)

    with progress:
        progress_task = progress.add_task(description, total=total)
        for value in sequence:
            yield value
            progress.advance(
                progress_task,
                advance=min(advance, total - progress.tasks[progress_task].completed),
            )
            progress.refresh()


def track_with_stats(
    sequence: Union[Sequence[ProgressType], Iterable[ProgressType]],
    stats: Dict[str, str],
    description: str = None,
    total: Optional[float] = None,
    advance: int = 1,
    transient: bool = False,
    console: Optional[Console] = None,
    disable: bool = False,
) -> Iterable[ProgressType]:
    """Track progress and displays stats by iterating over a sequence.

    Args:
        sequence (Iterable[ProgressType]): A sequence (must support "len") you wish to iterate over.
        stats (Dict[str, str]): A dictionary of stats to display.
        description (str, optional): Description of task show next to progress bar. Defaults to `None`.
        total (float, optional): Total number of steps. Default is len(sequence).
        advance (int, optional): Number of steps to advance progress by. Defaults to 1. Total / advance must less than or equal to len(sequence) for progress to reach finished state.
        transient (bool, optional): Clear the progress on exit. Defaults to False.
        console (Console, optional): Console to write to. Default creates internal Console instance.
        disable (bool, optional): Disable display of progress.
    Returns:
        Iterable[ProgressType]: An iterable of the values in the sequence.
    """
    progress = _autolabel_progress(
        description=description,
        transient=transient,
        console=console,
        disable=disable,
    )
    stats_progress = Progress(
        TextColumn("{task.fields[stats]}"),
    )

    group = Group(progress, stats_progress)
    live = Live(group)

    if total is None:
        total = len(sequence)

    with live:
        progress_task = progress.add_task(description=description, total=total)
        stats_task = stats_progress.add_task(
            "Stats", stats=", ".join(f"{k}={v}" for k, v in stats.items())
        )
        for value in sequence:
            yield value
            progress.advance(
                progress_task,
                advance=min(advance, total - progress.tasks[progress_task].completed),
            )
            stats_progress.update(
                stats_task, stats=", ".join(f"{k}={v}" for k, v in stats.items())
            )
            live.refresh()


def maybe_round(value: Any) -> Any:
    """Round's value only if it has a round function"""
    if hasattr(value, "__round__"):
        return round(value, 4)
    else:
        return value


def print_table(
    data: Dict,
    show_header: bool = True,
    console: Optional[Console] = None,
    default_style: str = "bold",
    styles: Dict = {},
) -> None:
    """Print a table of data.

    Args:
        data (Dict[str, List]): A dictionary of data to print.
        show_header (bool, optional): Show the header row. Defaults to True.
        console (Console, optional): Console to write to. Default creates internal Console instance.
        default_style (str, optional): Default style to apply to the table. Defaults to "bold".
        styles (Dict, optional): A dictionary of styles to apply to the table.
    """
    # Convert all values to strings
    data = {
        str(key): [str(maybe_round(v)) for v in value]
        if isinstance(value, List)
        else [str(maybe_round(value))]
        for key, value in data.items()
    }
    table = Table(show_header=show_header)
    for key in data:
        table.add_column(key, style=styles.get(key, default_style))
    for i, row in enumerate(zip(*data.values())):
        table.add_row(*row)
    console = console or Console()
    console.print(table)


def get_data(dataset_name: str, force: bool = False):
    """Download Datasets

    Args:
        dataset_name (str): dataset name
        force (bool, optional): if set to True, downloads and overwrites the local test and seed files
            if false then downloads onlyif the files are not present locally
    """

    def download_bar(current, total, width=80):
        """custom progress bar for downloading data"""
        width = shutil.get_terminal_size()[0] // 2
        print(
            f"{current//total*100}% [{'.' * (current//total * int(width))}] [{current}/{total}] bytes",
            end="\r",
        )

    def download(url: str) -> None:
        """Downloads the data given an url"""
        file_name = os.path.basename(url)
        if force and os.path.exists(file_name):
            print(f"File {file_name} exists. Removing")
            os.remove(file_name)

        if not os.path.exists(file_name):
            print(f"Downloading example dataset from {url} to {file_name}...")
            wget.download(url, bar=download_bar)

    if dataset_name not in EXAMPLE_DATASETS:
        logger.error(
            f"{dataset_name} not in list of available datasets: {str(EXAMPLE_DATASETS)}. Exiting..."
        )
        return
    seed_url = DATASET_URL.format(dataset=dataset_name, partition="seed")
    test_url = DATASET_URL.format(dataset=dataset_name, partition="test")
    try:
        if dataset_name not in NO_SEED_DATASET:
            download(seed_url)
        download(test_url)
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
