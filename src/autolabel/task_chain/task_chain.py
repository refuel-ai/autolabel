from collections import defaultdict
import copy
from itertools import accumulate, groupby
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
from pydantic.v1 import BaseModel
from autolabel.configs import TaskChainConfig
from autolabel.schema import TaskType
from autolabel.transforms import TransformFactory
from transformers import AutoTokenizer
import pandas as pd

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class TaskGraph:
    def __init__(self, task_chain: List[Dict]):
        self.graph = defaultdict(set)
        self.task_chain = task_chain

    def add_dependency(self, pre_task: str, post_task: str):
        """Add dependencies between pairs of tasks

        Args:
            pre_task (str): The task that must be completed before post_task
            post_task (str): The task that depends on pre_task
        """
        self.graph[pre_task].add(post_task)

    def topological_sort_helper(self, pre_task: str, visited: Dict, stack: List):
        """Recursive helper function to perform topological sort

        Args:
            pre_task (str): The task we are currently visiting
            visited (Dict): Dict of visited tasks
            stack (List): Stack to store the sorted tasks (in reverse order)
        """
        visited[pre_task] = True

        for post_task in self.graph[pre_task]:
            if visited[post_task] == False:
                self.topological_sort_helper(post_task, visited, stack)
        stack.append(pre_task)

    def topological_sort(self) -> List[str]:
        """Topological sort of the task graph

        Returns:
            List[str]: List of task names in topological order
        """
        visited = defaultdict(bool)
        stack = []

        for task in self.task_chain:
            if visited[task.get("task_name")] == False:
                self.topological_sort_helper(task.get("task_name"), visited, stack)
        return stack[::-1]

    def check_cycle(self):
        """Check for cycles in the task graph

        Returns:
            bool: True if cycle is present, False otherwise"""
        visited = defaultdict(bool)
        rec_stack = defaultdict(bool)

        for task in self.task_chain:
            if visited[task.get("task_name")] == False:
                if self.check_cycle_helper(task.get("task_name"), visited, rec_stack):
                    return True
        return False

    def check_cycle_helper(self, pre_task: str, visited: Dict, rec_stack: Dict):
        """Recursive helper function to check for cycles
        Args:
            pre_task (str): The task we are currently visiting
            visited (Dict): List of visited tasks
            rec_stack (Dict): A recursive tack to store the current path
        Returns:
            bool: True if cycle is present, False otherwise
        """
        visited[pre_task] = True
        rec_stack[pre_task] = True

        for post_task in self.graph[pre_task]:
            if visited[post_task] == False:
                if self.check_cycle_helper(post_task, visited, rec_stack) == True:
                    return True
            elif rec_stack[post_task] == True:
                return True
        rec_stack[pre_task] = False
        return False


class TaskChainOrchestrator:
    def __init__(
        self,
        task_chain_config: TaskChainConfig,
        cache: Optional[bool] = True,
        example_selector: Optional[BaseExampleSelector] = None,
        generation_cache: Optional[BaseCache] = SQLAlchemyGenerationCache(),
        transform_cache: Optional[BaseCache] = SQLAlchemyTransformCache(),
        confidence_cache: Optional[BaseCache] = SQLAlchemyConfidenceCache(),
        confidence_tokenizer: Optional[AutoTokenizer] = None,
        column_name_map: Optional[Dict[str, str]] = None,
    ):
        self.task_chain_config = task_chain_config
        self.cache = cache
        self.example_selector = example_selector
        self.generation_cache = generation_cache
        self.transform_cache = transform_cache
        self.confidence_cache = confidence_cache
        self.confidence_tokenizer = confidence_tokenizer
        self.column_name_map = column_name_map

    # TODO: For now, we run each separate step of the task chain serially and aggregate at the end.
    # We can optimize this with parallelization where possible/no dependencies.
    async def run(self, dataset_df: pd.DataFrame):
        """
        Run the different subtasks in the task chain on the input dataset

        Args:
            dataset_df (pd.DataFrame): Input dataset
        Returns:
            AutolabelDataset: Output dataset with the results of the task chain
        """
        subtasks = self.task_chain_config.subtasks()
        if len(subtasks) == 0:
            raise ValueError("No subtasks found in the task chain")
        for task in subtasks:
            autolabel_config = AutolabelConfig(task)
            dataset = AutolabelDataset(dataset_df, autolabel_config)
            if autolabel_config.transforms():
                agent = LabelingAgent(
                    config=autolabel_config,
                    cache=self.cache,
                    example_selector=self.example_selector,
                    generation_cache=self.generation_cache,
                    transform_cache=self.transform_cache,
                    confidence_cache=self.confidence_cache,
                    confidence_tokenizer=self.confidence_tokenizer,
                )
                for transform_dict in autolabel_config.transforms():
                    transform = TransformFactory.from_dict(
                        transform_dict,
                        cache=None,
                    )
                    dataset = await agent.async_run_transform(transform, dataset)
            else:
                agent = LabelingAgent(
                    config=task,
                    cache=self.cache,
                    example_selector=self.example_selector,
                    generation_cache=self.generation_cache,
                    transform_cache=self.transform_cache,
                    confidence_cache=self.confidence_cache,
                    confidence_tokenizer=self.confidence_tokenizer,
                )
                dataset = await agent.arun(
                    dataset,
                    skip_eval=True,
                )
            dataset = self.rename_output_columns(dataset, autolabel_config)
            dataset_df = dataset.df
        return dataset

    def rename_output_columns(
        self, dataset: AutolabelDataset, autolabel_config: AutolabelConfig
    ):
        """
        Rename the output columns of the dataset for each intermediate step in the task chain so that
        they are consistent with the expected input columns of future steps

        Args:
            dataset (AutolabelDataset): Input dataset
            task (ChainTask): The current task in the task chain
        Returns:
            AutolabelDataset: The dataset with renamed output columns
        """
        if autolabel_config.transforms():
            dataset.df.rename(columns=self.column_name_map, inplace=True)
        else:
            if autolabel_config.task_type() == TaskType.ATTRIBUTE_EXTRACTION:
                for attribute in autolabel_config.output_columns():
                    dataset.df[attribute] = dataset.df[
                        dataset.generate_label_name("label")
                    ].apply(
                        lambda x: x.get(attribute) if x and type(x) is dict else None
                    )
            elif autolabel_config.task_type() == TaskType.MULTILABEL_CLASSIFICATION:
                for output_column in autolabel_config.output_columns():
                    dataset.df[output_column] = dataset.df[
                        dataset.generate_label_name("label")
                    ]

        return dataset
