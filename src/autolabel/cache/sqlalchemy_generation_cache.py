from sqlalchemy.orm import sessionmaker
from typing import Optional
from autolabel.schema import GenerationCacheEntry
from autolabel.database import create_db_engine
from autolabel.data_models import Base
from .base import BaseCache
from typing import List, Union
from langchain.schema import Generation, ChatGeneration
from autolabel.data_models import GenerationCacheEntryModel
import logging

logger = logging.getLogger(__name__)


class SQLAlchemyGenerationCache(BaseCache):
    """A cache system implemented with SQL Alchemy"""

    def __init__(self):
        self.engine = None
        self.base = Base
        self.session = None

    def initialize(self):
        self.engine = create_db_engine()
        self.base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()

    def lookup(
        self, entry: GenerationCacheEntry
    ) -> List[Union[Generation, ChatGeneration]]:
        """Retrieves an entry from the Cache. Returns an empty list [] if not found.
        Args:
            entry: GenerationCacheEntry we wish to retrieve from the Cache
        Returns:
            result: A list of langchain Generation objects, containing the results of the labeling run for this GenerationCacheEntry. Empty list [] if not found.
        """
        cache_entry = GenerationCacheEntryModel.get(self.session, entry)
        if cache_entry is None:
            logger.debug("Cache miss")
            return []

        logger.debug("Cache hit")
        return cache_entry.generations

    def update(self, entry: GenerationCacheEntry) -> None:
        """Inserts the provided GenerationCacheEntry into the Cache, overriding it if it already exists
        Args:
            entry: GenerationCacheEntry we wish to put into the Cache
        """
        GenerationCacheEntryModel.insert(self.session, entry)

    def clear(self) -> None:
        """Clears the entire Cache"""
        GenerationCacheEntryModel.clear(self.session)
