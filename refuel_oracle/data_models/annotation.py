from .base import Base
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import Column, Integer, JSON
import json


class AnnotationModel(Base):
    __abstract__ = True  # this line is necessary

    id = Column(Integer, primary_key=True, autoincrement=True)
    index = Column(Integer)
    llm_annotation = Column(JSON)
    ANNOTATION_MODEL_CLASSES = {}

    def __repr__(self):
        return f"<AnnotationModel(id={self.id}, index={self.index}, annotation={self.llm_annotation})"

    # build a model class with a specific table name
    @classmethod
    def get_annotation_model(cls, task_result_id: int):
        tablename = f"annotations_{task_result_id}"  # dynamic table name
        class_name = f"AnnotationModel{task_result_id}"  # dynamic class name
        if class_name not in cls.ANNOTATION_MODEL_CLASSES:
            cls.ANNOTATION_MODEL_CLASSES[class_name] = type(
                class_name,
                (AnnotationModel,),
                {"__tablename__": tablename},
            )

        return cls.ANNOTATION_MODEL_CLASSES[class_name]

    @classmethod
    def get_annotation_table_name(cls, task_result_id: int):
        return f"annotations_{task_result_id}"

    @classmethod
    def create(cls, db, annotation: BaseModel):
        logger.debug(f"creating new annotation: {annotation}")
        db_object = cls(**json.loads(annotation.json()))
        db.add(db_object)
        db_object = db.query(cls).order_by(cls.id.desc()).first()
        logger.debug(f"created new annotation: {db_object}")
        return db_object

    @classmethod
    def from_pydantic(cls, annotation: BaseModel):
        return cls(**json.loads(annotation.json()))

    def delete(self, db):
        db.delete(self)
