from typing import List, Optional
import pickle

from langchain.cache import SQLAlchemyCache
from langchain.schema import Generation
from sqlalchemy.orm import Session
from sqlalchemy import select

from refuel_oracle.database import create_db_engine


class LLMCache(SQLAlchemyCache):
    """Cache LLM calls with sqlite as a backend. Caches both generations and metadata (e.g. logprobs)"""

    def __init__(self):
        engine = create_db_engine()
        super().__init__(engine)

    def lookup(self, prompt: str, llm_string: str) -> Optional[List[Generation]]:
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

    def update(
        self, prompt: str, llm_string: str, return_val: List[Generation]
    ) -> None:
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
