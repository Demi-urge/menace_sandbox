import pytest
from types import SimpleNamespace

import sync_shared_db

# Skip the test module if watchdog is unavailable
pytest.importorskip("watchdog.events")
watchdog_observers = pytest.importorskip("watchdog.observers")


def test_run_watch_triggers_sync_on_fs_events(tmp_path, monkeypatch):
    queue_dir = tmp_path
    queue_file = queue_dir / "q.jsonl"
    calls = []

    def fake_sync_once(qdir, conn):
        calls.append((qdir, conn))

    monkeypatch.setattr(sync_shared_db, "_sync_once", fake_sync_once)

    class FakeObserver:
        def __init__(self, *args, **kwargs):
            self.handler = None
            self.path = None

        def schedule(self, handler, path):
            self.handler = handler
            self.path = path

        def start(self):
            queue_file.write_text("a", encoding="utf-8")
            event = SimpleNamespace(is_directory=False, src_path=str(queue_file))
            self.handler.on_created(event)
            queue_file.write_text("ab", encoding="utf-8")
            self.handler.on_modified(event)

        def stop(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(watchdog_observers, "Observer", FakeObserver)

    sync_shared_db._run_watch(queue_dir, object(), interval=0, once=True)

    assert len(calls) == 3
    assert all(call[0] == queue_dir for call in calls)
