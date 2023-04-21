from refuel_oracle.data_models import (
    AnnotationModel,
    Base,
    DatasetModel,
    TaskModel,
    TaskResultModel,
)
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker


class Database:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.base = Base
        self.session = None

    def initialize(self):
        self.base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, autocommit=True)()

    def create_annotation_table(self, task_result_id: int):
        table_name = AnnotationModel.get_annotation_table_name(task_result_id)
        print(f"creating table {table_name} ...")
        self.base.metadata.reflect(self.engine)
        print(self.base.metadata.tables)
        table = self.base.metadata.tables.get(table_name)
        if table is not None:
            print(f"table {table_name} already exists")
            return table_name

        annotation_model = AnnotationModel.get_annotation_model(task_result_id)
        annotation_model.__table__.create(self.engine)
        # annotation_model.task_result = (
        #     "TaskResultModel", back_populates="annotations"
        # )
        # TaskResultModel.annotations = relationshrelationshipip(
        #     annotation_model, back_populates="task_result"
        # )
        print(f"table {table_name} created")
        return table_name

    def delete_annotation_table(self, task_result_id: int):
        table_name = AnnotationModel.get_annotation_table_name(task_result_id)
        self.base.metadata.reflect(self.engine)
        table = self.base.metadata.tables.get(table_name)
        if table is None:
            print(f"table {table_name} does not exist!")
            return

        print(f"deleting table {table_name} ...")
        table.drop(self.engine)
        self.base.metadata.remove(table)
        print(f"table {table_name} deleted!")
