from .base import Base
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
import json
from refuel_oracle.utils import calculate_md5
from refuel_oracle.dataset_config import DatasetConfig


class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(String(32), primary_key=True)
    input_file = Column(String(50))
    start_index = Column(Integer)
    end_index = Column(Integer)
    task_results = relationship("TaskResultModel", back_populates="dataset")

    def __repr__(self):
        return f"<DatasetModel(id={self.id}, input_file={self.input_file}, start_index={self.start_index}, end_index={self.end_index})>"

    @classmethod
    def create_id(
        self,
        input_file: str,
        dataset_config: DatasetConfig,
        start_index: int,
        max_items: int,
    ):
        filehash = calculate_md5(
            [open(input_file, "rb"), dataset_config.dict, start_index, max_items]
        )
        return filehash

    @classmethod
    def create(cls, db, dataset: BaseModel):
        db_object = cls(**json.loads(dataset.json()))
        db.add(db_object)
        return db_object

    @classmethod
    def get_by_id(cls, db, id: int):
        return db.query(cls).filter(cls.id == id).first()

    @classmethod
    def get_by_input_file(cls, db, input_file: str):
        return db.query(cls).filter(cls.input_file == input_file).first()

    def delete(self, db):
        db.delete(self)
