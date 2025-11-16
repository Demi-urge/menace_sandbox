import types
import sandbox_runner.environment as env


def test_overlay_cleanup_fallback_records_events(monkeypatch, tmp_path):
    o1 = tmp_path / "vm1"
    o2 = tmp_path / "vm2"
    o1.mkdir()
    o2.mkdir()
    (o1 / "overlay.qcow2").touch()
    (o2 / "overlay.qcow2").touch()

    overlays = tmp_path / "overlays.json"
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_FILE", overlays)
    monkeypatch.setattr(env, "_ACTIVE_OVERLAYS_LOCK", env.FileLock(str(overlays) + ".lock"))
    monkeypatch.setattr(env, "_FAILED_OVERLAYS_FILE", tmp_path / "failed.json")
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", tmp_path / "failed_cleanup.json")
    monkeypatch.setattr(env.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(env, "psutil", None)
    monkeypatch.setattr(env, "_OVERLAY_MAX_AGE", 0.0)
    monkeypatch.setattr(env, "_increment_cleanup_stat", lambda *a, **k: None)

    env._write_active_overlays([str(o1)])

    events = []
    monkeypatch.setattr(env, "_write_cleanup_log", lambda e: events.append(e))

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        if cmd[:2] == ["pgrep", "-fa"]:
            line = f"123 qemu-system-x86_64 file={o2 / 'overlay.qcow2'}\n"
            return types.SimpleNamespace(returncode=0, stdout=line, stderr="")
        if cmd[0] == "kill":
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    removed = env._purge_stale_vms()

    assert removed >= 2
    assert not o1.exists()
    assert not o2.exists()
    assert env._read_active_overlays() == []

    event_map = {(e["resource_id"], e["reason"]): e["success"] for e in events}
    assert event_map.get((str(o1), "vm_overlay")) is True
    assert event_map.get((str(o2), "vm_overlay")) is True
    assert event_map.get(("123", "vm_process")) is True
