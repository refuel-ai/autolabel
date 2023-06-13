from sqlalchemy.orm import sessionmaker
import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd

from autolabel.data_models import Base
from autolabel.data_models import DatasetModel, TaskModel, TaskRunModel
from autolabel.schema import Dataset, Task, TaskRun, TaskStatus
from autolabel.configs import AutolabelConfig

from .engine import create_db_engine

logger = logging.getLogger(__name__)


class StateManager:
    def __init__(self):
        self.engine = create_db_engine()
        self.base = Base
        self.session = None

    def initialize(self):
        self.base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, autocommit=True)()

    def initialize_dataset(
        self,
        dataset: Union[str, pd.DataFrame],
        config: AutolabelConfig,
        start_index: int,
        max_items: Optional[int],
    ):
        # TODO: Check if this works for max_items = None

        dataset_id = Dataset.create_id(dataset, config, start_index, max_items)
        dataset_orm = DatasetModel.get_by_id(self.session, dataset_id)
        if dataset_orm:
            return Dataset.from_orm(dataset_orm)

        dataset = Dataset(
            id=dataset_id,
            input_file=dataset if isinstance(dataset, str) else "",
            start_index=start_index,
            end_index=start_index + max_items if max_items else -1,
        )
        return Dataset.from_orm(DatasetModel.create(self.session, dataset))

    def initialize_task(self, config: AutolabelConfig):
        task_id = Task.create_id(config)
        task_orm = TaskModel.get_by_id(self.session, task_id)
        if task_orm:
            return Task.from_orm(task_orm)

        task = Task(
            id=task_id,
            config=config.to_json(),
            task_type=config.task_type(),
            provider=config.provider(),
            model_name=config.model_name(),
        )
        return Task.from_orm(TaskModel.create(self.session, task))

    def get_task_run(self, task_id: str, dataset_id: str):
        task_run_orm = TaskRunModel.get(self.session, task_id, dataset_id)
        if task_run_orm:
            return TaskRun.from_orm(task_run_orm)
        else:
            return None

    def create_task_run(
        self, output_file: str, task_id: str, dataset_id: str
    ) -> TaskRun:
        logger.debug(f"creating new task_run")
        new_task_run = TaskRun(
            task_id=task_id,
            dataset_id=dataset_id,
            status=TaskStatus.ACTIVE,
            current_index=0,
            output_file=output_file,
            created_at=datetime.now(),
        )
        task_run_orm = TaskRunModel.create(self.session, new_task_run)
        return TaskRun.from_orm(task_run_orm)
