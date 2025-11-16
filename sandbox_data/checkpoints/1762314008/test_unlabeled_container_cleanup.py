import types
import importlib
import sys
from pathlib import Path

import dynamic_path_router
from dynamic_path_router import resolve_path


def test_unlabeled_container_removed(monkeypatch, tmp_path):
    # prepare environment module with minimal dependencies
    class DummyErrorLogger:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setitem(
        sys.modules, "error_logger", types.SimpleNamespace(ErrorLogger=DummyErrorLogger)
    )
    monkeypatch.setattr(
        dynamic_path_router,
        "resolve_path",
        lambda name: Path(name),
    )
    resolve_path = dynamic_path_router.resolve_path
    env = importlib.import_module("sandbox_runner.environment")
    importlib.reload(env)

    # ensure active files do not interfere
    monkeypatch.setattr(env, '_read_active_containers', lambda: [])
    monkeypatch.setattr(env, '_write_active_containers', lambda ids: None)
    monkeypatch.setattr(env, '_read_active_overlays', lambda: [])
    monkeypatch.setattr(env, '_purge_stale_vms', lambda: 0)
    monkeypatch.setattr(env, '_PRUNE_VOLUMES', False)
    monkeypatch.setattr(env, '_PRUNE_NETWORKS', False)
    monkeypatch.setattr(env, '_CONTAINER_MAX_LIFETIME', 0.0)

    cmds = []

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False):
        cmds.append(cmd)
        if cmd[:4] == ['docker', 'ps', '-aq', f'--filter']:
            return types.SimpleNamespace(returncode=0, stdout='')
        if cmd[:3] == ['docker', 'ps', '-a']:
            out = (
                f'abc\t2023-01-01 00:00:00 +0000 UTC\t"python {resolve_path("sandbox_runner.py").name}"\n'  # path-ignore
                'def\t2023-01-01 00:00:00 +0000 UTC\t"other"\n'
            )
            return types.SimpleNamespace(returncode=0, stdout=out)
        if cmd[:3] == ['docker', 'rm', '-f']:
            return types.SimpleNamespace(returncode=0, stdout='')
        return types.SimpleNamespace(returncode=0, stdout='')

    monkeypatch.setattr(env.subprocess, 'run', fake_run)

    env._STALE_CONTAINERS_REMOVED = 0
    env.purge_leftovers()

    assert ['docker', 'rm', '-f', 'abc'] in cmds
    assert ['docker', 'rm', '-f', 'def'] not in cmds
    assert env._STALE_CONTAINERS_REMOVED == 1
