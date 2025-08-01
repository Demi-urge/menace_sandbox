import asyncio
import multiprocessing
import pytest
import os
import types
import sys

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///"))
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
pydantic_mod = types.ModuleType("pydantic")
pydantic_dc = types.ModuleType("dataclasses")
pydantic_dc.dataclass = lambda *a, **k: (lambda cls: cls)
pydantic_mod.dataclasses = pydantic_dc
pydantic_mod.BaseModel = object
sys.modules.setdefault("pydantic", pydantic_mod)
sys.modules.setdefault("pydantic.dataclasses", pydantic_dc)
import importlib.util
import importlib.machinery
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_self_test_service():
    if 'menace' in sys.modules:
        del sys.modules['menace']
    pkg = types.ModuleType('menace')
    pkg.__path__ = [str(ROOT)]
    pkg.__spec__ = importlib.machinery.ModuleSpec('menace', loader=None, is_package=True)
    sys.modules['menace'] = pkg
    spec = importlib.util.spec_from_file_location('menace.self_test_service', ROOT / 'self_test_service.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['menace.self_test_service'] = mod
    spec.loader.exec_module(mod)
    return mod


sts = load_self_test_service()


def test_container_locking(monkeypatch):
    async def fake_exec(*cmd, **kwargs):
        if cmd[0] == 'docker' and cmd[1] == '--version':
            class P:
                returncode = 0
                async def wait(self):
                    return None
            return P()
        elif cmd[0] == 'docker' and cmd[1] == 'ps':
            class P:
                returncode = 0
                async def communicate(self):
                    return b'', b''
                async def wait(self):
                    return None
            return P()
        elif cmd[0] == 'docker' and cmd[1] == 'rm':
            class P:
                returncode = 0
                async def wait(self):
                    return None
            return P()
        elif cmd[0] == 'docker':
            class P:
                returncode = 0
                async def communicate(self):
                    await asyncio.sleep(0.1)
                    return json.dumps({'summary': {'passed': 0, 'failed': 0}}).encode(), b''
                async def wait(self):
                    return None
            return P()
        else:
            raise RuntimeError('unexpected command')

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake_exec)

    svc1 = sts.SelfTestService(use_container=True)
    svc2 = sts.SelfTestService(use_container=True)

    async def run_svc(svc):
        await svc._run_once()

    async def main():
        start = time.perf_counter()
        await asyncio.gather(
            asyncio.create_task(run_svc(svc1)),
            asyncio.create_task(run_svc(svc2)),
        )
        return time.perf_counter() - start

    elapsed = asyncio.run(main())
    assert elapsed >= 0.19


def test_container_retries(monkeypatch):
    calls = []

    async def fake_exec(*cmd, **kwargs):
        if cmd[0] == 'docker' and cmd[1] == '--version':
            class P:
                returncode = 0

                async def wait(self):
                    return None

            return P()
        elif cmd[0] == 'docker' and cmd[1] == 'ps':
            class P:
                returncode = 0

                async def communicate(self):
                    return b'', b''

                async def wait(self):
                    return None

            return P()
        elif cmd[0] == 'docker' and cmd[1] == 'rm':
            class P:
                returncode = 0

                async def wait(self):
                    return None

            return P()
        elif cmd[0] == 'docker':
            calls.append('run')

            if len(calls) == 1:
                class P:
                    returncode = 1
                    stdout = asyncio.StreamReader()

                    async def communicate(self):
                        self.stdout.feed_data(b'')
                        self.stdout.feed_eof()
                        return b'', b''

                    async def wait(self):
                        return None

                return P()
            else:
                class P:
                    returncode = 0
                    stdout = asyncio.StreamReader()

                    async def communicate(self):
                        self.stdout.feed_data(json.dumps({'summary': {'passed': 0, 'failed': 0}}).encode())
                        self.stdout.feed_eof()
                        return b'', b''

                    async def wait(self):
                        return None

                return P()
        else:
            raise RuntimeError('unexpected command')

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake_exec)

    svc = sts.SelfTestService(use_container=True, container_retries=1)
    svc.run_once()
    assert calls.count('run') == 2


def _proc_run_once(root: str, lock_path: str, q):
    import os
    import time
    import importlib.util
    import importlib.machinery
    import types
    import json
    import asyncio
    import sys

    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
    os.environ["SELF_TEST_LOCK_FILE"] = lock_path

    pkg = types.ModuleType('menace')
    pkg.__path__ = [root]
    pkg.__spec__ = importlib.machinery.ModuleSpec('menace', loader=None, is_package=True)
    sys.modules['menace'] = pkg
    spec = importlib.util.spec_from_file_location('menace.self_test_service', Path(root) / 'self_test_service.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['menace.self_test_service'] = mod
    spec.loader.exec_module(mod)

    async def fake_exec(*cmd, **kwargs):
        if cmd[0] == 'docker' and cmd[1] == '--version':
            class P:
                returncode = 0

                async def wait(self):
                    return None

            return P()
        elif cmd[0] == 'docker' and cmd[1] == 'ps':
            class P:
                returncode = 0

                async def communicate(self):
                    return b'', b''

                async def wait(self):
                    return None

            return P()
        elif cmd[0] == 'docker' and cmd[1] == 'rm':
            class P:
                returncode = 0

                async def wait(self):
                    return None

            return P()
        elif cmd[0] == 'docker':
            q.put((os.getpid(), 'start', time.time()))

            class P:
                returncode = 0

                async def communicate(self):
                    await asyncio.sleep(0.2)
                    q.put((os.getpid(), 'end', time.time()))
                    return json.dumps({'summary': {'passed': 0, 'failed': 0}}).encode(), b''

                async def wait(self):
                    return None

            return P()
        else:
            raise RuntimeError('unexpected command')

    mod.asyncio.create_subprocess_exec = fake_exec
    svc = mod.SelfTestService(use_container=True)
    svc.run_once()


def test_file_lock_across_processes(tmp_path):
    lock_path = str(tmp_path / 'svc.lock')
    q = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=_proc_run_once, args=(str(ROOT), lock_path, q))
    p2 = multiprocessing.Process(target=_proc_run_once, args=(str(ROOT), lock_path, q))
    p1.start()
    time.sleep(0.05)
    p2.start()
    p1.join()
    p2.join()

    records = [q.get() for _ in range(4)]
    grouped: dict[int, list[tuple[str, float]]] = {}
    for pid, name, t in records:
        grouped.setdefault(pid, []).append((name, t))

    (start1, end1), (start2, end2) = [sorted(v) for v in grouped.values()]
    assert start2[1] >= end1[1]
