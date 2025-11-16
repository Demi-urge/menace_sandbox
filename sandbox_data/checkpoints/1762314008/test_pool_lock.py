import multiprocessing
import time
import asyncio
import pytest
import importlib.util
import importlib.machinery
import types
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _proc_lock(root: str, lock_path: str, q):
    import os
    import asyncio
    import atexit

    atexit.register = lambda *a, **k: None
    os.environ["SANDBOX_POOL_LOCK"] = lock_path

    pkg = types.ModuleType('sandbox_runner')
    pkg.__path__ = [str(Path(root)/'sandbox_runner')]
    pkg.__spec__ = importlib.machinery.ModuleSpec('sandbox_runner', loader=None, is_package=True)
    sys.modules['sandbox_runner'] = pkg
    spec = importlib.util.spec_from_file_location('sandbox_runner.environment', Path(root)/'sandbox_runner'/'environment.py')  # path-ignore
    env = importlib.util.module_from_spec(spec)
    sys.modules['sandbox_runner.environment'] = env
    spec.loader.exec_module(env)

    async def run():
        async with env.pool_lock():
            q.put((os.getpid(), 'start', time.time()))
            await asyncio.sleep(0.2)
            q.put((os.getpid(), 'end', time.time()))
    asyncio.run(run())


def test_pool_lock_across_processes(tmp_path):
    lock_path = str(tmp_path / 'pool.lock')
    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=_proc_lock, args=(str(ROOT), lock_path, q))
    p2 = multiprocessing.Process(target=_proc_lock, args=(str(ROOT), lock_path, q))
    p1.start()
    time.sleep(0.05)
    p2.start()
    p1.join()
    p2.join()

    records = [q.get() for _ in range(4)]
    grouped = {}
    for pid, name, t in records:
        grouped.setdefault(pid, []).append((name, t))

    (start1, end1), (start2, end2) = [sorted(v) for v in grouped.values()]
    assert start2[1] >= end1[1]


def _proc_fail(root: str, lock_path: str, q, mode: str):
    import os
    import asyncio
    import atexit

    atexit.register = lambda *a, **k: None
    os.environ["SANDBOX_POOL_LOCK"] = lock_path

    pkg = types.ModuleType('sandbox_runner')
    pkg.__path__ = [str(Path(root)/'sandbox_runner')]
    pkg.__spec__ = importlib.machinery.ModuleSpec('sandbox_runner', loader=None, is_package=True)
    sys.modules['sandbox_runner'] = pkg
    spec = importlib.util.spec_from_file_location('sandbox_runner.environment', Path(root)/'sandbox_runner'/'environment.py')  # path-ignore
    env = importlib.util.module_from_spec(spec)
    sys.modules['sandbox_runner.environment'] = env
    spec.loader.exec_module(env)

    class Failing:
        def run(self, *a, **k):
            raise RuntimeError("boom")

    class DummyClient:
        def __init__(self):
            self.containers = Failing()

    env._DOCKER_CLIENT = DummyClient()
    env._log_pool_metrics = lambda *a, **k: None

    async def run_create():
        try:
            await env._create_pool_container("img")
        except Exception:
            pass

    async def run_get():
        try:
            await env._get_pooled_container("img")
        except Exception:
            pass

    async def main():
        if mode == 'create':
            await run_create()
        else:
            await run_get()
        q.put(os.path.exists(lock_path))
        os._exit(0)

    asyncio.run(main())


def test_lock_removed_on_create_failure(tmp_path):
    lock_path = str(tmp_path / 'pool.lock')
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_proc_fail, args=(str(ROOT), lock_path, q, 'create'))
    p.start()
    p.join()
    assert q.get() is False


def test_lock_removed_on_get_failure(tmp_path):
    lock_path = str(tmp_path / 'pool.lock')
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_proc_fail, args=(str(ROOT), lock_path, q, 'get'))
    p.start()
    p.join()
    assert q.get() is False
