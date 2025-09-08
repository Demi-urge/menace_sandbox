import asyncio
import json
import importlib.util
import sys
from pathlib import Path

# Ensure menace package can be imported similar to other tests
ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("menace", ROOT / "__init__.py")  # path-ignore
menace_pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(menace_pkg)
sys.modules.setdefault("menace", menace_pkg)

import menace.self_test_service as sts


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        if k.get("return_metadata"):
            return "", {}
        return ""

async def _dummy_proc(results: dict[str, int]):
    class P:
        returncode = 0
        stdout = asyncio.StreamReader()

        async def wait(self):
            self.stdout.feed_data(json.dumps({"summary": results}).encode())
            self.stdout.feed_eof()
            return None

    return P()


def test_container_env_vars(monkeypatch):
    recorded = {}

    async def fake_exec(*cmd, **kwargs):
        recorded["cmd"] = cmd
        return await _dummy_proc({"passed": 0, "failed": 0})

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def avail(self):
        return True

    monkeypatch.setattr(sts.SelfTestService, "_docker_available", avail)
    monkeypatch.setenv("TEST_ENV_VAR", "42")

    svc = sts.SelfTestService(use_container=True, container_image="img", context_builder=DummyBuilder())
    svc.run_once()

    cmd = recorded["cmd"]
    assert cmd[0] == "docker"
    found = False
    for i in range(len(cmd) - 1):
        if cmd[i] == "-e" and str(cmd[i + 1]) == "TEST_ENV_VAR=42":
            found = True
            break
    assert found


def test_podman_offline_load(monkeypatch, tmp_path):
    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(cmd)
        return await _dummy_proc({"passed": 0, "failed": 0})

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def avail(self):
        return True

    monkeypatch.setattr(sts.SelfTestService, "_docker_available", avail)

    tar_path = tmp_path / "img.tar"
    tar_path.write_text("dummy")
    monkeypatch.setenv("MENACE_OFFLINE_INSTALL", "1")
    monkeypatch.setenv("MENACE_SELF_TEST_IMAGE_TAR", str(tar_path))

    svc = sts.SelfTestService(
        use_container=True,
        container_image="img",
        container_runtime="podman",
        docker_host="ssh://host",
        context_builder=DummyBuilder(),
    )
    svc.run_once()

    load_call = next((c for c in calls if "load" in c), None)
    assert load_call and load_call[0] == "podman"
    run_call = next((c for c in calls if "run" in c), None)
    assert run_call and "--url" in run_call and "ssh://host" in run_call
