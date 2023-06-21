from sqlalchemy.orm import sessionmaker
from typing import Optional
from autolabel.schema import CacheEntry
from autolabel.database import create_db_engine
from autolabel.data_models import Base
from autolabel.cache import BaseCache
from typing import List
from langchain.schema import Generation
from autolabel.data_models import CacheEntryModel
import logging

logger = logging.getLogger(__name__)


class SQLAlchemyCache(BaseCache):
    """A cache system implemented with SQL Alchemy"""

    def __init__(self):
        self.engine = create_db_engine()
        self.base = Base
        self.session = None
        self.initialize()

    def initialize(self):
        self.base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine, autocommit=True)()

    def lookup(self, entry: CacheEntry) -> List[Generation]:
        """Retrieves an entry from the Cache. Returns an empty list [] if not found.
        Args:
            entry: CacheEntry we wish to retrieve from the Cache
        Returns:
            result: A list of langchain Generation objects, containing the results of the labeling run for this CacheEntry. Empty list [] if not found.
        """
        cache_entry = CacheEntryModel.get(self.session, entry)
        if cache_entry is None:
            logger.debug("Cache miss")
            return []

        logger.debug("Cache hit")
        return cache_entry.generations

    def update(self, entry: CacheEntry) -> None:
        """Inserts the provided CacheEntry into the Cache, overriding it if it already exists
        Args:
            entry: CacheEntry we wish to put into the Cache
        """
        CacheEntryModel.insert(self.session, entry)

    def clear(self) -> None:
        """Clears the entire Cache"""
        CacheEntryModel.clear(self.session)
