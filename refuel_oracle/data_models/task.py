from .base import Base
from pydantic import BaseModel
from sqlalchemy import Column, String, Text
from sqlalchemy.orm import relationship
import json

from refuel_oracle.task_config import TaskConfig
from refuel_oracle.models import ModelConfig
from refuel_oracle.utils import calculate_md5


class TaskModel(Base):
    __tablename__ = "tasks"

    id = Column(String(32), primary_key=True)
    task_type = Column(String(50))
    provider = Column(String(50))
    model_name = Column(String(50))
    config = Column(Text)
    task_results = relationship("TaskResultModel", back_populates="task")

    def __repr__(self):
        return f"<TaskModel(id={self.id}, task_type={self.task_type}, provider={self.provider}, model_name={self.model_name})>"

    @classmethod
    def create_id(self, task_config: TaskConfig, llm_config: ModelConfig):
        filehash = calculate_md5([task_config.config, llm_config.dict])
        return filehash

    @classmethod
    def create(cls, db, task: BaseModel):
        db_object = cls(**json.loads(task.json()))
        db.add(db_object)
        return db_object

    @classmethod
    def get_by_id(cls, db, id: int):
        return db.query(cls).filter(cls.id == id).first()

    def delete(self, db):
        db.delete(self)
