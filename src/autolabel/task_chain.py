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
    ExampleSelectorFactory,
    BaseExampleSelector,
    DEFAULT_EMBEDDING_PROVIDER,
    PROVIDER_TO_MODEL,
)
from autolabel.cache.sqlalchemy_generation_cache import SQLAlchemyGenerationCache
from autolabel.cache.sqlalchemy_transform_cache import SQLAlchemyTransformCache
from autolabel.cache.sqlalchemy_confidence_cache import SQLAlchemyConfidenceCache
from autolabel.cache.base import BaseCache
from pydantic import BaseModel
from autolabel.configs import TaskChainConfig
from autolabel.schema import TaskType
from autolabel.transforms import TransformFactory
from transformers import AutoTokenizer
import pandas as pd

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class ChainTask(BaseModel):
    type: str
    id: str
    name: str
    input_columns: List[str]
    output_columns: List[str]
    autolabel_config: AutolabelConfig = None
    config: Dict = None

    class Config:
        arbitrary_types_allowed = True


class TaskGraph:
    def __init__(self, task_chain: List[ChainTask]):
        self.graph = defaultdict(set)
        self.task_chain = task_chain

    def add_dependency(self, pre_task: str, post_task: str):
        self.graph[pre_task].add(post_task)

    def topological_sort_helper(self, pre_task: str, visited: List, stack: List):
        """Recursive helper function to perform topological sort"""
        visited[pre_task] = True

        for post_task in self.graph[pre_task]:
            if visited[post_task] == False:
                self.topological_sort_helper(post_task, visited, stack)
        stack.append(pre_task)

    def topological_sort(self) -> List[str]:
        visited = defaultdict(bool)
        stack = []

        for task in self.task_chain:
            if visited[task.id] == False:
                self.topological_sort_helper(task.id, visited, stack)
        return stack[::-1]


class TaskChainOrchestrator:
    def __init__(
        self,
        task_chain_config: TaskChainConfig,
        dataset_df: pd.DataFrame,
        cache: Optional[bool] = True,
        example_selector: Optional[BaseExampleSelector] = None,
        generation_cache: Optional[BaseCache] = SQLAlchemyGenerationCache(),
        transform_cache: Optional[BaseCache] = SQLAlchemyTransformCache(),
        confidence_cache: Optional[BaseCache] = SQLAlchemyConfidenceCache(),
        confidence_tokenizer: Optional[AutoTokenizer] = None,
        column_name_map: Optional[Dict[str, str]] = None,
    ):
        self.task_chain_config = task_chain_config
        self.task_configs = self.initialize_task_configs()
        self.dataset_df = dataset_df
        self.cache = cache
        self.example_selector = example_selector
        self.generation_cache = generation_cache
        self.transform_cache = transform_cache
        self.confidence_cache = confidence_cache
        self.confidence_tokenizer = confidence_tokenizer
        self.column_name_map = column_name_map
        self.attributes = []  # TODO REMOVE
        self.task_chain: List[ChainTask] = self.initialize_task_chain()
        logger.info(f"task chain: {self.task_chain}")
        self.task_graph = self.initialize_task_graph()
        logger.info(f"task graph: {self.task_graph.graph}")
        self.sort_task_chain()

    def initialize_task_configs(self):
        task_configs = []
        subtasks = self.task_chain_config.subtasks()
        for input_columns, group in groupby(subtasks, lambda x: x.get("input_columns")):
            task_config = copy.deepcopy(self.task_chain_config.config)
            task_config["task_name"] = task_config["task_name"]
            task_config["task_type"] = TaskType.ATTRIBUTE_EXTRACTION
            example_template = ""
            input_columns = [column.lower() for column in input_columns]
            for column in input_columns:
                example_template += f"\n{column.capitalize()}: { {column} }".replace(
                    "'", ""
                )
            example_template += "\nOutput: {output_dict}"
            example_template = example_template.strip()
            task_config["dataset"]["input_columns"] = input_columns
            task_config["prompt"]["example_template"] = example_template
            task_config["prompt"]["attributes"] = list(group)
            task_configs.append(AutolabelConfig(task_config))
        logger.info(f"task_configs: {task_configs}")
        return task_configs

    def initialize_task_chain(self):
        task_chain = []
        for task_config in self.task_configs:
            output_columns = [
                attribute.get("name") for attribute in task_config.attributes()
            ]
            self.attributes.extend(output_columns)
            task_chain.append(
                ChainTask(
                    type="label",
                    id=str(uuid.uuid4()),
                    name=task_config.task_name(),
                    input_columns=task_config.input_columns(),
                    output_columns=output_columns,
                    autolabel_config=task_config,
                    config=task_config.config,
                )
            )
            for transform in task_config.transforms():
                task_chain.append(
                    ChainTask(
                        type="transform",
                        id=str(uuid.uuid4()),
                        name=transform.get("name"),
                        input_columns=transform.get("input_columns", []),
                        output_columns=list(
                            transform.get("output_columns", {}).values()
                        ),
                        autolabel_config=task_config,
                        config=transform,
                    )
                )
        return task_chain

    def initialize_task_graph(self):
        task_graph = TaskGraph(self.task_chain)
        chain_output_columns = {}
        for task in self.task_chain:
            for output_column in task.output_columns:
                chain_output_columns[output_column] = task.id
        for task in self.task_chain:
            for input_column in task.input_columns:
                if input_column in chain_output_columns:
                    pre_task = chain_output_columns[input_column]
                    task_graph.add_dependency(pre_task, task.id)
        return task_graph

    def sort_task_chain(self):
        task_order = self.task_graph.topological_sort()
        logger.info(f"Task Order: {task_order}")
        self.task_chain = sorted(
            self.task_chain, key=lambda task: task_order.index(task.id)
        )

    async def run(self):
        if not self.task_chain:
            raise ValueError("No task configurations provided")
        dataset = AutolabelDataset(self.dataset_df, self.task_chain[0].autolabel_config)
        for task in self.task_chain:
            logger.info(f"Running task: {task.id}")
            dataset = AutolabelDataset(dataset.df, task.autolabel_config)
            if task.type == "label":
                agent = LabelingAgent(
                    config=task.autolabel_config,
                    cache=False,
                    example_selector=self.example_selector,
                    generation_cache=self.generation_cache,
                    transform_cache=self.transform_cache,
                    confidence_cache=self.confidence_cache,
                    confidence_tokenizer=self.confidence_tokenizer,
                )
                logger.info(f"dataset df columns 3: {len(dataset.df.columns)}")
                dataset = await agent.arun(
                    dataset,
                    skip_eval=True,
                )
                logger.info(f"dataset df columns 4: {len(dataset.df.columns)}")
            elif task.type == "transform":
                transform = TransformFactory.from_dict(
                    task.config,
                    cache=None,
                )
                agent = LabelingAgent(
                    config=task.autolabel_config,
                    cache=self.cache,
                    example_selector=self.example_selector,
                    generation_cache=None,
                    transform_cache=None,
                    confidence_cache=None,
                    confidence_tokenizer=self.confidence_tokenizer,
                )
                logger.info(f"dataset df columns 1: {dataset.df.columns}")
                dataset = await agent.async_run_transform(transform, dataset)
                logger.info(f"dataset df columns 2: {dataset.df.columns}")
            dataset = self.aggregate(dataset, task)
        dataset = self.clean(dataset)
        return dataset

    def aggregate(self, dataset: AutolabelDataset, task: ChainTask):
        if task.type == "label":
            for attribute in task.output_columns:
                dataset.df[attribute] = dataset.df[
                    dataset.generate_label_name("label")
                ].apply(lambda x: x.get(attribute) if x and type(x) is dict else None)
        elif task.type == "transform":
            dataset.df.rename(columns=self.column_name_map, inplace=True)
        logger.info(f"dataset df columns 5: {dataset.df.columns}")
        return dataset

    def create_dict(self, row):
        d = {}
        for attr in self.attributes:
            d[attr] = row[attr]
        logger.info(f"d: {d}")
        return d

    def clean(self, dataset: AutolabelDataset):
        # dataset.df[dataset.generate_label_name("label")] = dataset.df[
        #     dataset.generate_label_name("label")
        # ].apply(lambda x: {attr: dataset.df[attr] for attr in self.attributes})
        dataset.df[dataset.generate_label_name("label")] = dataset.df.apply(
            self.create_dict, axis=1
        )
        return dataset
