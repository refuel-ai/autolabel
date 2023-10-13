from sqlalchemy.orm import sessionmaker
from typing import Optional
from autolabel.schema import ConfidenceCacheEntry
from autolabel.database import create_db_engine
from autolabel.data_models import Base
from .base import BaseCache
from autolabel.data_models import ConfidenceCacheEntryModel
import logging

logger = logging.getLogger(__name__)


class SQLAlchemyConfidenceCache(BaseCache):
    """A cache system implemented with SQL Alchemy"""

    def __init__(self):
        self.engine = create_db_engine()
        self.base = Base
        self.session = None

    def initialize(self):
        self.base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()

    def lookup(self, entry: ConfidenceCacheEntry) -> float:
        """Retrieves an entry from the Cache. Returns None if not found.
        Args:
            entry: ConfidenceCacheEntry we wish to retrieve from the Cache
        Returns:
            result: A floating point number representing the confidence score for this generation. None if not found.
        """
        cache_entry = ConfidenceCacheEntryModel.get(self.session, entry)
        if cache_entry is None:
            logger.debug("Cache miss")
            return None

        logger.debug("Cache hit")
        return cache_entry.logprobs

    def update(self, entry: ConfidenceCacheEntry) -> None:
        """Inserts the provided ConfidenceCacheEntry into the Cache, overriding it if it already exists
        Args:
            entry: ConfidenceCacheEntry we wish to put into the Cache
        """
        ConfidenceCacheEntryModel.insert(self.session, entry)

    def clear(self) -> None:
        """Clears the entire Cache"""
        ConfidenceCacheEntryModel.clear(self.session)
