from refuel_oracle.data_models import Base
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from loguru import logger
from typing import Tuple
from refuel_oracle.data_models import DatasetModel, TaskModel, TaskResultModel
from refuel_oracle.dataset_config import DatasetConfig
from refuel_oracle.schema import Dataset, Task, TaskResult, TaskStatus
from refuel_oracle.task_config import TaskConfig
from refuel_oracle.llm import LLMConfig


class Database:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.base = Base
        self.session = None

    def initialize(self):
        self.base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, autocommit=True)()

    def initialize_dataset(
        self, input_file: str, dataset_config: DatasetConfig, start_index, max_items
    ):
        # TODO: Check if this works for max_items = None
        dataset_id = DatasetModel.create_id(
            input_file, dataset_config, start_index, max_items
        )
        dataset_orm = DatasetModel.get_by_id(self.session, dataset_id)
        if dataset_orm:
            return Dataset.from_orm(dataset_orm)

        dataset = Dataset(
            id=dataset_id,
            input_file=input_file,
            start_index=start_index,
            end_index=start_index + max_items,
        )
        return Dataset.from_orm(DatasetModel.create(self.session, dataset))

    def initialize_task(self, task_config: TaskConfig, llm_config: LLMConfig):
        task_id = TaskModel.create_id(task_config, llm_config)
        task_orm = TaskModel.get_by_id(self.session, task_id)
        if task_orm:
            return Task.from_orm(task_orm)

        task = Task(
            id=task_id,
            config=self.task_config.to_json(),
            task_type=self.task_config.get_task_type(),
            provider=self.llm_config.get_provider(),
            model_name=self.llm_config.get_model_name(),
        )
        return Task.from_orm(TaskModel.create(self.db.session, task))

    def initialize_task_result(
        self, output_file, task_object, dataset
    ) -> Tuple[TaskResult, bool]:
        task_result_orm = TaskResultModel.get(self.session, task_object.id, dataset.id)
        task_result = TaskResult.from_orm(task_result_orm) if task_result_orm else None
        if task_result:
            logger.debug(f"existing task_result: {task_result}")
            return task_result, True
        else:
            logger.debug(f"creating new task_result")
            new_task_result = TaskResult(
                task_id=task_object.id,
                dataset_id=dataset.id,
                status=TaskStatus.ACTIVE,
                current_index=0,
                output_file=output_file,
            )
            task_result_orm = TaskResultModel.create(self.db.session, new_task_result)
            return TaskResult.from_orm(task_result_orm), False
