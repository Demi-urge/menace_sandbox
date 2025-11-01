from __future__ import annotations

"""Scaffold helper for embeddable databases.

This script creates a new SQLite-backed module that mixes in
:class:`menace_sandbox.embeddable_db_mixin.EmbeddableDBMixin`, wires up a basic
full-text-search schema based on ``sql_templates/create_fts.sql``,
and installs simple license and safety hooks.
"""

import argparse
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
from typing import Any, Iterator, List
from pathlib import Path
import sqlite3
import logging

from menace_sandbox.embeddable_db_mixin import EmbeddableDBMixin
from license_detector import detect as detect_license
from unsafe_patterns import find_matches as find_unsafe
from dynamic_path_router import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class {record_class}:
    data: str
    rec_id: int = 0


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
        self.safety_check(rec)
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

    def safety_check(self, rec: {record_class} | dict[str, Any]) -> None:
        text = rec.data if isinstance(rec, {record_class}) else rec.get("data", "")
        issues = find_unsafe(text)
        if issues:
            raise ValueError("; ".join(issues))
'''


def _camel(name: str) -> str:
    return ''.join(part.capitalize() for part in name.split('_'))


def create_db_scaffold(name: str, root: Path = Path('.')) -> Path:
    """Create a new embeddable database module in *root*.

    The function writes ``<name>_db.py`` and updates ``__init__.py`` to
    import the generated class and expose the module via ``__all__``.
    """

    snake = name.lower()
    class_name = _camel(snake) + 'DB'
    record_class = _camel(snake) + 'Record'
    module_name = f'{snake}_db'

    mod_path = root / f'{module_name}.py'
    mod_path.write_text(
        MODULE_TEMPLATE.format(
            class_name=class_name,
            record_class=record_class,
            snake=snake,
        )
    )

    init_path = root / '__init__.py'
    if init_path.exists():
        text = init_path.read_text()
        import_line = f'from .{module_name} import {class_name}\n'
        if import_line not in text:
            text += '\n' + import_line
        marker = '__all__ = ['
        if marker in text and f'"{module_name}"' not in text:
            idx = text.index(marker) + len(marker)
            text = text[:idx] + f'\n    "{module_name}",' + text[idx:]
        init_path.write_text(text)

    return mod_path


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Scaffold a new database')
    parser.add_argument('name', help='Base name for the database')
    args = parser.parse_args(argv)
    create_db_scaffold(args.name, root=Path.cwd())
    print(f'Created scaffold for {args.name}')
    return 0


if __name__ == '__main__':  # pragma: no cover - manual use
    raise SystemExit(cli())
