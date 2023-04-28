from typing import Optional, Type
from langchain.cache import (
    SQLAlchemyCache,
    RETURN_VAL_TYPE,
)
from sqlalchemy.orm import Session
from sqlalchemy import select, create_engine
import pickle


class RefuelSQLLangchainCache(SQLAlchemyCache):
    """Cache that extends the SQAlchemy as a backend."""

    def __init__(self, database_path: str = ".langchain.db"):
        engine = create_engine(f"sqlite:///{database_path}")
        super().__init__(engine)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        stmt = (
            select(self.cache_schema.response)
            .where(self.cache_schema.prompt == prompt)
            .where(self.cache_schema.llm == llm_string)
            .order_by(self.cache_schema.idx)
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).fetchall()
            if rows:
                return [pickle.loads(row[0]) for row in rows]
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update based on prompt and llm_string."""
        items = [
            self.cache_schema(
                prompt=prompt, llm=llm_string, response=pickle.dumps(gen), idx=i
            )
            for i, gen in enumerate(return_val)
        ]
        with Session(self.engine) as session, session.begin():
            for item in items:
                session.merge(item)
