from typing import Optional
from autolabel.schema import CacheEntry
from database import create_db_engine
from autolabel.cache import BaseCache
from typing import List
from langchain.schema import Generation
from autolabel.data_models import CacheEntryModel


class SQLAlchemyCache(BaseCache):
    def __init__(self):
        self.engine = create_db_engine()

    def lookup(self, entry: CacheEntry) -> List[Generation]:
        cache_entry = CacheEntryModel.get(self.engine, entry)
        if cache_entry is None:
            return []
        return cache_entry.generations
