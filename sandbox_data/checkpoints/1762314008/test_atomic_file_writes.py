import json
import sandbox_runner.environment as env
import pytest

@pytest.mark.parametrize("attr,write_func", [
    ("_ACTIVE_CONTAINERS_FILE", env._write_active_containers),
    ("_ACTIVE_OVERLAYS_FILE", env._write_active_overlays),
    ("_FAILED_OVERLAYS_FILE", env._write_failed_overlays),
    ("FAILED_CLEANUP_FILE", env._write_failed_cleanup),
])
def test_atomic_write_no_corruption(monkeypatch, tmp_path, attr, write_func):
    target = tmp_path / "target.json"
    target.write_text(json.dumps(["old"]))
    monkeypatch.setattr(env, attr, target)
    if attr == "_ACTIVE_CONTAINERS_FILE":
        monkeypatch.setattr(env, "_ACTIVE_CONTAINERS_LOCK", env.FileLock(str(target) + ".lock"))
    if attr == "_ACTIVE_OVERLAYS_FILE":
        monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(target) + ".lock"))

    tmp = tmp_path / "tmp.json"
    monkeypatch.setattr(env.tempfile, "NamedTemporaryFile", lambda *a, **k: open(tmp, "w+", encoding="utf-8"))
    monkeypatch.setattr(env.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))

    write_func(["new"])  # should not raise

    assert json.loads(target.read_text()) == ["old"]
    assert tmp.exists()
