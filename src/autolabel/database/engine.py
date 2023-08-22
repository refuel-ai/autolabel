from typing import Any, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine

DB_ENGINE = None

# This creates one global ".autolabel.db" in your home directory.
# Having one global SQLite database in ~ is a poor idea, since
# SQLite cannot handle multiple simultaneous writes (if you have
# several different labeling jobs going on).
# DB_PATH = join(expanduser("~"), ".autolabel.db")

DB_PATH = ".autolabel.v15.db"


def create_db_engine(db_path: Optional[str] = DB_PATH) -> Engine:
    global DB_ENGINE
    if DB_ENGINE is None:
        DB_ENGINE = create_engine(f"sqlite:///{db_path}", pool_size=0)
    return DB_ENGINE
