from collections import defaultdict
import copy
from itertools import groupby
import uuid
from autolabel.configs import AutolabelConfig
import logging
from typing import Dict, List, Optional

from autolabel.labeler import LabelingAgent
from autolabel.dataset import AutolabelDataset
from autolabel.few_shot import (
    BaseExampleSelector,
)
from autolabel.cache.sqlalchemy_generation_cache import SQLAlchemyGenerationCache
from autolabel.cache.sqlalchemy_transform_cache import SQLAlchemyTransformCache
from autolabel.cache.sqlalchemy_confidence_cache import SQLAlchemyConfidenceCache
from autolabel.cache.base import BaseCache
from pydantic import BaseModel
from autolabel.configs import TaskChainConfig
from autolabel.schema import TASK_CHAIN_TYPE, TaskType
from autolabel.transforms import TransformFactory
from transformers import AutoTokenizer
import pandas as pd
from autolabel.task_chain import TaskChainOrchestrator, TaskGraph

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


# TODO: allow this to take in a list of dicts or autolabel configs and infer accordingly
def initialize_task_chain_config(task_chain_name: str, configs: List[Dict]) -> Dict:
    """Initialize the task chain config

    Args:
        task_chain_name (str): The name of the task chain
        configs (List[AutolabelConfig]): The list of autolabel configs

    Returns:
        Dict: The task chain config
    """
    task_graph = initialize_task_graph(configs)
    subtasks = sort_subtasks(configs, task_graph)
    return {
        "task_name": task_chain_name,
        "task_type": TASK_CHAIN_TYPE,
        "subtasks": subtasks,
    }


def initialize_task_graph(subtasks: List[Dict]) -> TaskGraph:
    """
    Create a task graph to represent the dependencies between tasks in the task chain

    Args:
        subtasks (List[Dict]): The list of subtasks in the task chain
    Returns:
        TaskGraph: A task graph object representing the dependencies between tasks
    """
    task_graph = TaskGraph(subtasks)
    chain_output_columns = {}
    for subtask in subtasks:
        autolabel_task = AutolabelConfig(subtask)
        for output_column in autolabel_task.output_columns():
            chain_output_columns[output_column] = subtask.get("task_name")
    for subtask in subtasks:
        autolabel_task = AutolabelConfig(subtask)
        for input_column in autolabel_task.input_columns():
            if input_column in chain_output_columns:
                pre_task = chain_output_columns[input_column]
                task_graph.add_dependency(pre_task, subtask.get("task_name"))
    return task_graph


def sort_subtasks(subtasks: List[Dict], task_graph: TaskGraph) -> List[Dict]:
    """Sort subtasks in topological order
    Args:
        subtasks (List[Dict]): The list of unsorted subtasks in the task chain
        task_graph (TaskGraph): The task graph representing the dependencies between tasks
    Returns:
        List[Dict]: The sorted subtasks in the task chain
    """
    task_order = task_graph.topological_sort()
    return sorted(subtasks, key=lambda task: task_order.index(task.get("task_name")))


# TODO: we should also validate that the subtasks are indeed sorted in topological order
def validate_task_chain(task_chain_config: TaskChainConfig) -> bool:
    """Validate the task graph by checking for cycles

    Returns:
        bool: True if the graph is valid, False otherwise
    """
    task_graph = initialize_task_graph(task_chain_config.subtasks())
    return not task_graph.check_cycle()
