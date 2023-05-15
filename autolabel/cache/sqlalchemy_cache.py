from sqlalchemy.orm import sessionmaker
from typing import Optional
from autolabel.schema import CacheEntry
from autolabel.database import create_db_engine
from autolabel.data_models import Base
from autolabel.cache import BaseCache
from typing import List
from langchain.schema import Generation
from autolabel.data_models import CacheEntryModel


class SQLAlchemyCache(BaseCache):
    def __init__(self):
        self.engine = create_db_engine()
        self.base = Base
        self.session = None
        self.initialize()

    def initialize(self):
        self.base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, autocommit=True)()

    def lookup(self, entry: CacheEntry) -> List[Generation]:
        cache_entry = CacheEntryModel.get(self.session, entry)
        if cache_entry is None:
            return []
        return cache_entry.generations

    def update(self, entry: CacheEntry):
        CacheEntryModel.insert(self.session, entry)
