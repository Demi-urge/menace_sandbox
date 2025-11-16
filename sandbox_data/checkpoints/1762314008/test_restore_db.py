import importlib.util
import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

from dynamic_path_router import resolve_path

ROOT = resolve_path(".")
sys.path.insert(0, str(ROOT))


def _setup_stubs(tmp_path):
    sql_spec = importlib.util.spec_from_file_location(
        "neurosales.sql_db", str(resolve_path("neurosales/sql_db.py"))
    )
    sql_db = importlib.util.module_from_spec(sql_spec)
    sys.modules["neurosales.sql_db"] = sql_db
    sql_spec.loader.exec_module(sql_db)

    pkg_root = tmp_path / "stub"
    pkg = pkg_root / "neurosales"
    pkg.mkdir(parents=True)
    with open(resolve_path("neurosales/sql_db.py"), "rb") as src:
        with open(pkg / ("sql_db" + ".py"), "wb") as dst:
            dst.write(src.read())
    init_file = (pkg / "__init__").with_suffix(".py")
    init_file.write_text(
        "from .sql_db import Base, create_session, UserProfile\n"
    )

    sys.modules["neurosales"] = types.ModuleType("neurosales")
    sys.modules["neurosales"].sql_db = sql_db

    backup_spec = importlib.util.spec_from_file_location(
        "neurosales.db_backup", str(resolve_path("neurosales/db_backup.py"))
    )
    db_backup = importlib.util.module_from_spec(backup_spec)
    sys.modules["neurosales.db_backup"] = db_backup
    backup_spec.loader.exec_module(db_backup)

    return sql_db, db_backup, pkg_root


def test_restore_db(tmp_path):
    sql_db, db_backup, pkg_root = _setup_stubs(tmp_path)
    create_session = sql_db.create_session
    UserProfile = sql_db.UserProfile

    os.environ.pop("NEURO_DB_POOL_SIZE", None)
    os.environ.pop("NEURO_DB_MAX_OVERFLOW", None)

    def fixed_backup_database(backup_path="backup.json", db_url=None):
        db_url = db_url or os.getenv("NEURO_DB_URL", "sqlite://")
        out = Path(backup_path)
        engine = sql_db.sa.create_engine(db_url)
        sql_db.Base.metadata.create_all(engine, checkfirst=True)
        Session = sql_db.sa.orm.sessionmaker(bind=engine)
        insp = sql_db.sa.inspect(engine)
        data = {}
        with Session() as s:
            for table in insp.get_table_names():
                rows = [dict(r._mapping) for r in s.execute(sql_db.sa.text(f"SELECT * FROM {table}"))]
                data[table] = rows
        out.write_text(importlib.import_module('json').dumps(data, indent=2))

    db_backup.backup_database = fixed_backup_database

    db_file = tmp_path / "db.sqlite"
    url = f"sqlite:///{db_file}"
    Session = create_session(url)
    with Session() as s:
        s.add(UserProfile(id="u1", username="alice"))
        s.commit()

    backup = tmp_path / "backup.json"
    db_backup.backup_database(str(backup), db_url=url)

    db_file.unlink()

    env = os.environ.copy()
    env["NEURO_DB_URL"] = url
    env["PYTHONPATH"] = f"{pkg_root}:{ROOT}"
    subprocess.check_call(
        [sys.executable, str(resolve_path("scripts/restore_db.py")), str(backup)],
        env=env,
        cwd=ROOT,
    )

    Session2 = create_session(url)
    with Session2() as s:
        user = s.get(UserProfile, "u1")
        assert user is not None
        assert user.username == "alice"


def test_restore_requires_env(tmp_path):
    env = os.environ.copy()
    env["NEURO_DB_URL"] = ""
    result = subprocess.run(
        [
            sys.executable,
            str(resolve_path("scripts/migrate.py")),
            "upgrade",
        ],
        env=env,
        cwd=ROOT,
        capture_output=True,
    )
    assert result.returncode != 0


def teardown_module(module):
    for name in ["neurosales.db_backup", "neurosales.sql_db", "neurosales"]:
        sys.modules.pop(name, None)
    import importlib
    importlib.import_module("neurosales.api_gateway")
