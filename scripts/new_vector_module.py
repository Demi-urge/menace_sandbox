from __future__ import annotations

"""Scaffold helper for vector_service database modules.

This utility creates a module under ``vector_service/`` (or a specified
location) that subclasses :class:`menace_sandbox.embeddable_db_mixin.EmbeddableDBMixin` and
includes context extraction, indexing setup, license checks, and basic
governance hooks.  It can also (optionally) register the module with
``db_router`` and create an initial Alembic migration.
"""

import argparse
import sys
import time
from pathlib import Path
import os
import uuid

from db_router import init_db_router
from dynamic_path_router import resolve_path

MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)

MODULE_TEMPLATE = '''from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List
import sqlite3
import logging

from menace_sandbox.embeddable_db_mixin import EmbeddableDBMixin
from license_detector import detect as detect_license
from security.secret_redactor import redact_dict
from dynamic_path_router import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class {record_class}:
    data: str
    rec_id: int = 0


def build_context(rec: {record_class} | dict[str, Any]) -> str:
    """Return a redacted context snippet for *rec*."""
    data = rec.data if isinstance(rec, {record_class}) else rec.get("data", "")
    return redact_dict({{"text": data}})["text"]


class {class_name}(EmbeddableDBMixin):
    """SQLite-backed store for {snake} records."""

    def __init__(
        self,
        path: str = "{snake}.db",
        *,
        vector_index_path: str = "{snake}.index",
        embedding_version: int = 1,
        vector_backend: str = "annoy",
    ) -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        EmbeddableDBMixin.__init__(
            self,
            index_path=vector_index_path,
            metadata_path=Path(vector_index_path).with_suffix(".json"),
            embedding_version=embedding_version,
            backend=vector_backend,
        )
        schema = resolve_path("sql_templates/create_fts.sql").read_text()
        schema = schema.replace("code_fts", "{snake}_fts")
        self.conn.executescript(schema)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS {snake}(
                rec_id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, rec: {record_class}) -> int:
        self.license_check(rec)
        rec = self.safety_check(rec)
        cur = self.conn.execute(
            "INSERT INTO {snake}(data) VALUES(?)",
            (rec.data,),
        )
        self.conn.commit()
        rec_id = int(cur.lastrowid)
        self._embed_record_on_write(rec_id, rec)
        return rec_id

    def _embed_record_on_write(self, rec_id: int, rec: {record_class} | dict[str, Any]) -> None:
        try:
            self.add_embedding(rec_id, rec, "{snake}", source_id=str(rec_id))
        except Exception:  # pragma: no cover - best effort
            logger.exception("embedding hook failed for %s", rec_id)

    def iter_records(self) -> Iterator[tuple[int, dict[str, Any], str]]:
        cur = self.conn.execute("SELECT rec_id, data FROM {snake}")
        for rec_id, data in cur.fetchall():
            yield rec_id, {{"data": data}}, "{snake}"

    def vector(self, rec: {record_class} | dict[str, Any] | str) -> List[float] | None:
        if isinstance(rec, {record_class}):
            text = rec.data
        elif isinstance(rec, dict):
            text = rec.get("data", "")
        else:
            text = str(rec)
        return self._embed(text) if text else None

    def license_check(self, rec: {record_class} | dict[str, Any]) -> None:
        text = rec.data if isinstance(rec, {record_class}) else rec.get("data", "")
        if detect_license(text):
            raise ValueError("disallowed license detected")

    def safety_check(self, rec: {record_class} | dict[str, Any]) -> {record_class} | dict[str, Any]:
        data = rec.data if isinstance(rec, {record_class}) else rec.get("data", "")
        redacted = redact_dict({{"data": data}})
        if isinstance(rec, {record_class}):
            rec.data = redacted["data"]
            return rec
        rec["data"] = redacted["data"]
        return rec
'''

MIGRATION_TEMPLATE = '''"""Initial table for {snake}."""

from alembic import op
import sqlalchemy as sa

revision = '{revision}'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        '{snake}',
        sa.Column('rec_id', sa.Integer, primary_key=True),
        sa.Column('data', sa.Text),
    )


def downgrade():
    op.drop_table('{snake}')
'''


def _camel(name: str) -> str:
    return ''.join(part.capitalize() for part in name.split('_'))


def create_scaffold(
    name: str,
    *,
    root: Path,
    register_router: bool = False,
    create_migration: bool = False,
) -> Path:
    """Create a new vector_service module scaffold."""

    snake = name.lower()
    class_name = _camel(snake) + 'DB'
    record_class = _camel(snake) + 'Record'
    module_name = f'{snake}_db'

    mod_path = root / f'{module_name}.py'
    if mod_path.exists():
        print(f'Warning: {mod_path} already exists', file=sys.stderr)
        return mod_path

    root.mkdir(parents=True, exist_ok=True)
    mod_path.write_text(
        MODULE_TEMPLATE.format(
            class_name=class_name,
            record_class=record_class,
            snake=snake,
        )
    )

    init_path = root / '__init__.py'
    text = init_path.read_text() if init_path.exists() else ''
    import_line = (
        f'from .{module_name} import {class_name}, {record_class}, build_context\n'
    )
    if import_line not in text:
        text += '\n' + import_line
    marker = '__all__ = ['
    if marker in text and f'"{module_name}"' not in text:
        idx = text.index(marker) + len(marker)
        text = text[:idx] + f'\n    "{module_name}",' + text[idx:]
    elif marker not in text:
        text += '\n__all__ = [\n    "{module_name}",\n]\n'
    init_path.write_text(text)

    if register_router:
        _register_in_router(module_name, class_name, record_class)
    if create_migration:
        _create_migration(snake)

    return mod_path


def _register_in_router(module_name: str, class_name: str, record_class: str) -> None:
    path = resolve_path('db_router.py')
    if not path.exists():
        return
    text = path.read_text()
    import_line = (
        f'from .vector_service.{module_name} import {class_name}, {record_class}\n'
    )
    if import_line not in text:
        anchor = 'from .bot_database import BotDB, BotRecord\n'
        if anchor in text:
            text = text.replace(anchor, anchor + import_line)
        else:
            text = import_line + text
    param_line = f'        {module_name}: {class_name} | None = None,\n'
    if param_line not in text:
        text = text.replace(
            '        workflow_db: Optional[WorkflowDB] = None,\n',
            '        workflow_db: Optional[WorkflowDB] = None,\n' + param_line,
        )
    assign_line = f'        self.{module_name} = {module_name} or {class_name}()\n'
    if assign_line not in text:
        text = text.replace(
            '        self.workflow_db = workflow_db or WorkflowDB(event_bus=event_bus)\n',
            '        self.workflow_db = workflow_db or WorkflowDB(event_bus=event_bus)\n'
            + assign_line,
        )
    path.write_text(text)


def _create_migration(snake: str) -> None:
    versions = Path('migrations') / 'versions'
    versions.mkdir(parents=True, exist_ok=True)
    rev = str(int(time.time()))
    mig_path = versions / f'{rev}_create_{snake}.py'
    if mig_path.exists():
        print(f'Warning: migration {mig_path} already exists', file=sys.stderr)
        return
    mig_path.write_text(MIGRATION_TEMPLATE.format(revision=rev, snake=snake))


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Scaffold a vector_service module')
    parser.add_argument('name', help='Base name for the module')
    parser.add_argument('--root', default='vector_service', help='Target directory')
    parser.add_argument('--register-router', action='store_true', help='Update db_router')
    parser.add_argument('--create-migration', action='store_true', help='Create alembic migration')
    args = parser.parse_args(argv)

    create_scaffold(
        args.name,
        root=Path(args.root),
        register_router=args.register_router,
        create_migration=args.create_migration,
    )
    print(f'Created scaffold for {args.name}')
    return 0


if __name__ == '__main__':  # pragma: no cover - manual execution
    raise SystemExit(cli())
