from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool
from logging.config import fileConfig
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
fileConfig(config.config_file_name)

db_url = os.getenv("DATABASE_URL", "sqlite:///menace.db")
config.set_main_option("sqlalchemy.url", db_url)

from menace.databases import MenaceDB  # type: ignore
from sqlalchemy import MetaData

if os.getenv("MENACE_SKIP_CREATE"):
    _orig = MetaData.create_all
    MetaData.create_all = lambda self, engine=None, **kw: None
    target_metadata = MenaceDB(db_url).meta
    MetaData.create_all = _orig
else:
    target_metadata = MenaceDB(db_url).meta

def run_migrations_offline() -> None:
    context.configure(url=db_url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    connectable = engine_from_config(config.get_section(config.config_ini_section), prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
