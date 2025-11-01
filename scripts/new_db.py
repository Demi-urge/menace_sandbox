"""Utility for scaffolding new ``EmbeddableDBMixin`` databases."""

from __future__ import annotations

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
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS {snake} (
                rec_id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, rec: {record_class}) -> int:
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

    def safety_check(self, rec: {record_class} | dict[str, Any]) -> None:
        """Placeholder safety check hook."""
        pass
'''


TEST_TEMPLATE = '''from {module_name} import {class_name}, {record_class}


def test_scaffold(tmp_path):
    db = {class_name}(path=tmp_path / "{snake}.db", vector_index_path=tmp_path / "{snake}.index")
    db._embed = lambda text: [0.0]
    rec = {record_class}(data="hello")
    rid = db.add(rec)
    rows = list(db.iter_records())
    assert rows and rows[0][0] == rid
'''


def _camel(name: str) -> str:
    return ''.join(part.capitalize() for part in name.split('_'))


def create_db_scaffold(name: str, root: Path = Path('.')) -> None:
    """Create module, test, and registration for a new database."""

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

    tests_dir = root / 'tests'
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_path = tests_dir / f'test_{module_name}.py'
    test_path.write_text(
        TEST_TEMPLATE.format(
            module_name=module_name,
            class_name=class_name,
            record_class=record_class,
            snake=snake,
        )
    )

    emb_file = resolve_path("vector_service/embedding_backfill.py")
    if emb_file.exists():
        text = emb_file.read_text()
        if module_name not in text:
            text = text.replace(
                'modules = [',
                f'modules = [\n            "{module_name}",',
                1,
            )
            emb_file.write_text(text)


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Scaffold a new database')
    parser.add_argument('name', help='Base name for the database')
    args = parser.parse_args(argv)
    create_db_scaffold(args.name)
    print(f'Created scaffold for {args.name}')
    return 0


if __name__ == '__main__':  # pragma: no cover - manual use
    raise SystemExit(cli())
