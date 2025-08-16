import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.db_backup import schedule_database_backup
from neurosales.sql_db import create_session as create_sql_session, UserProfile

os.environ.pop("NEURO_DB_POOL_SIZE", None)
os.environ.pop("NEURO_DB_MAX_OVERFLOW", None)


def test_schedule_database_backup(tmp_path):
    backup_file = tmp_path / "backup.json"
    Session = create_sql_session("sqlite://")
    # create a simple record
    with Session() as s:
        s.add(UserProfile(id="1", username="u"))
        s.commit()

    schedule_database_backup(interval=0.1, backup_path=str(backup_file), db_url="sqlite://")
    time.sleep(0.2)

    assert backup_file.exists()


def test_gateway_triggers_backup(monkeypatch, tmp_path):
    monkeypatch.setenv("NEURO_API_KEY", "secret")
    monkeypatch.setenv("NEURO_BACKUP_INTERVAL", "1")
    monkeypatch.setenv("NEURO_DB_URL", "sqlite://")
    backup_path = tmp_path / "auto.json"
    monkeypatch.setenv("NEURO_BACKUP_PATH", str(backup_path))

    called = {}

    def fake_schedule(interval=3600, *, backup_path="backup.json", db_url=None):
        called["interval"] = interval
        called["backup_path"] = backup_path
        called["db_url"] = db_url
        class Dummy:
            pass
        return Dummy()

    import neurosales.api_gateway  # ensure module is loaded
    monkeypatch.setattr(
        "neurosales.api_gateway.schedule_database_backup", fake_schedule
    )

    from neurosales.api_gateway import create_app

    create_app()

    assert called == {
        "interval": 1,
        "backup_path": str(backup_path),
        "db_url": "sqlite://",
    }

