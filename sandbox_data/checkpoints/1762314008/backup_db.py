import json
import os
from dotenv import load_dotenv
import shutil
import subprocess
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from neurosales.sql_db import Base

load_dotenv()


def dump_sqlalchemy(engine, path: Path) -> None:
    Session = sessionmaker(bind=engine)
    insp = sa.inspect(engine)
    data = {}
    with Session() as s:
        for table in insp.get_table_names():
            rows = [dict(r) for r in s.execute(sa.text(f"SELECT * FROM {table}"))]
            data[table] = rows
    path.write_text(json.dumps(data, indent=2))


def main() -> None:
    db_url = os.getenv("NEURO_DB_URL", "sqlite://")
    out = Path("backup.json")

    if db_url.startswith("postgresql") and shutil.which("pg_dump"):
        subprocess.check_call(["pg_dump", db_url, "-f", str(out)])
        print(f"Backup created with pg_dump at {out}")
        return

    engine = sa.create_engine(db_url)
    Base.metadata.create_all(engine, checkfirst=True)
    dump_sqlalchemy(engine, out)
    print(f"Backup saved to {out}")


if __name__ == "__main__":
    main()
