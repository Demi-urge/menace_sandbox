import asyncio
import contextlib
import json
import types
import sandbox_runner.environment as env


def _run_cleanup_once():
    async def runner():
        task = asyncio.create_task(env._cleanup_worker())
        await asyncio.sleep(0.02)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    asyncio.run(runner())


def test_cleanup_worker_removes_container(monkeypatch, tmp_path):
    cid = "c1"
    file = tmp_path / "failed.json"
    stats_file = tmp_path / "stats.json"
    file.write_text(json.dumps({cid: 0.0}))
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)
    monkeypatch.setattr(env, "_CLEANUP_STATS_FILE", stats_file)
    monkeypatch.setattr(env, "_POOL_CLEANUP_INTERVAL", 0.01)
    monkeypatch.setattr(env, "_cleanup_idle_containers", lambda *_, **__: (0, 0))
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=True: 0)

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        if cmd[:3] == ["docker", "rm", "-f"]:
            return types.SimpleNamespace(returncode=0, stdout="")
        if cmd[:2] == ["docker", "ps"]:
            return types.SimpleNamespace(returncode=0, stdout="")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    _run_cleanup_once()

    assert json.loads(file.read_text()) == {}


def test_cleanup_worker_removes_overlay(monkeypatch, tmp_path):
    overlay = tmp_path / "ov"
    overlay.mkdir()
    (overlay / "overlay.qcow2").touch()
    file = tmp_path / "failed.json"
    stats_file = tmp_path / "stats.json"
    file.write_text(json.dumps({str(overlay): 0.0}))
    monkeypatch.setattr(env, "FAILED_CLEANUP_FILE", file)
    monkeypatch.setattr(env, "_CLEANUP_STATS_FILE", stats_file)
    monkeypatch.setattr(env, "_POOL_CLEANUP_INTERVAL", 0.01)
    monkeypatch.setattr(env, "_cleanup_idle_containers", lambda *_, **__: (0, 0))
    monkeypatch.setattr(env, "_purge_stale_vms", lambda record_runtime=True: 0)

    removed = []

    def fake_rmtree(path):
        removed.append(path)
    monkeypatch.setattr(env.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=""))

    _run_cleanup_once()

    assert json.loads(file.read_text()) == {}
    assert str(overlay) in removed

