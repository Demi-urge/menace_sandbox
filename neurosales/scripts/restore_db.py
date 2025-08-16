import json
import os
from dotenv import load_dotenv
import shutil
import subprocess
import sys
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from neurosales.sql_db import Base

load_dotenv()


def _load_sqlalchemy(engine, path: Path) -> None:
    """Load table data from ``path`` using SQLAlchemy."""
    data = json.loads(path.read_text())
    metadata = sa.MetaData()
    metadata.reflect(bind=engine)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        for table_name, rows in data.items():
            if not rows:
                continue
            table = metadata.tables.get(table_name)
            if table is None:
                continue
            s.execute(table.insert(), rows)
        s.commit()


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("backup.json")
    db_url = os.getenv("NEURO_DB_URL", "sqlite://")

    if db_url.startswith("postgresql") and shutil.which("psql"):
        first = src.read_text(64).lstrip()
        if first.startswith("--"):
            subprocess.check_call(["psql", db_url, "-f", str(src)])
            print(f"Database restored from {src}")
            return

    engine = sa.create_engine(db_url)
    Base.metadata.create_all(engine, checkfirst=True)
    _load_sqlalchemy(engine, src)
    print(f"Data loaded from {src}")


if __name__ == "__main__":
    main()
