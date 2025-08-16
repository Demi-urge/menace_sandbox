import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional
import json

import sqlalchemy as sa

from .sql_db import Base


def backup_database(backup_path: str = "backup.json", db_url: Optional[str] = None) -> None:
    """Create a backup of the database at ``backup_path``.

    Parameters
    ----------
    backup_path:
        File path for the backup output. Defaults to ``backup.json``.
    db_url:
        Database connection string. If omitted, ``NEURO_DB_URL`` or an
        in-memory SQLite database is used.
    """
    db_url = db_url or os.getenv("NEURO_DB_URL", "sqlite://")
    out = Path(backup_path)

    if db_url.startswith("postgresql") and shutil.which("pg_dump"):
        subprocess.check_call(["pg_dump", db_url, "-f", str(out)])
        print(f"Backup created with pg_dump at {out}")
        return

    engine = sa.create_engine(db_url)
    Base.metadata.create_all(engine, checkfirst=True)

    Session = sa.orm.sessionmaker(bind=engine)
    insp = sa.inspect(engine)
    data = {}
    with Session() as s:
        for table in insp.get_table_names():
            rows = [dict(r) for r in s.execute(sa.text(f"SELECT * FROM {table}"))]
            data[table] = rows
    out.write_text(json.dumps(data, indent=2))
    print(f"Backup saved to {out}")


def schedule_database_backup(
    interval: int = 3600,
    *,
    backup_path: str = "backup.json",
    db_url: Optional[str] = None,
) -> threading.Thread:
    """Run ``backup_database`` periodically in a background thread."""

    def _loop() -> None:
        while True:
            backup_database(backup_path=backup_path, db_url=db_url)
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


__all__ = ["backup_database", "schedule_database_backup"]
