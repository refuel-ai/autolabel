from typing import Any, Optional
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from os.path import join, expanduser

DB_ENGINE = None
DB_PATH = join(expanduser("~"), ".autolabel.db")


def create_db_engine(db_path: Optional[str] = DB_PATH) -> Engine:
    global DB_ENGINE
    if DB_ENGINE is None:
        DB_ENGINE = create_engine(f"sqlite:///{db_path}")
    return DB_ENGINE
