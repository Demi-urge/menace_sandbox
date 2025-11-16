import asyncio
import os
import shutil
import subprocess
import pytest

import sandbox_runner.environment as env
from metrics_exporter import container_creation_success_total


def _docker_available():
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False
    try:
        res = subprocess.run([docker_bin, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except Exception:
        return False


@pytest.mark.integration
def test_real_docker_container(monkeypatch):
    pytest.importorskip("docker")
    if not _docker_available():
        pytest.skip("docker service unavailable")

    monkeypatch.setenv("SANDBOX_CONTAINER_POOL_SIZE", "0")
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()

    image = os.getenv("SANDBOX_CONTAINER_IMAGE", "python:3.11-slim")
    before = container_creation_success_total.labels(image=image).get()
    before_ids = set(env._CONTAINER_DIRS)

    metrics = asyncio.run(env._execute_in_container("print('ok')", {}))
    assert metrics.get("exit_code") == 0

    created_ids = set(env._CONTAINER_DIRS) - before_ids
    env._cleanup_pools()
    for cid in created_ids:
        assert cid not in env._CONTAINER_DIRS

    after = container_creation_success_total.labels(image=image).get()
    assert after >= before + 1


@pytest.mark.integration
def test_real_qemu_vm(tmp_path, monkeypatch):
    qemu = shutil.which("qemu-system-x86_64")
    if not qemu:
        pytest.skip("qemu-system-x86_64 not found")
    img = tmp_path / "base.qcow2"
    res = subprocess.run([
        shutil.which("qemu-img") or "qemu-img",
        "create",
        "-f",
        "qcow2",
        str(img),
        "10M",
    ])
    if res.returncode != 0 or not img.exists():
        pytest.skip("unable to create qcow2 image")

    monkeypatch.setenv("SANDBOX_DOCKER", "0")
    before = set(env._read_active_overlays())

    tracker = env.simulate_full_environment(
        {"OS_TYPE": "windows", "VM_SETTINGS": {"windows_image": str(img), "memory": "256M"}}
    )
    assert tracker.diagnostics.get("vm_error") is None
    assert set(env._read_active_overlays()) == before
