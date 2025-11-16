import os
import sys
import subprocess
import time
import shutil
import importlib.util

from dynamic_path_router import resolve_path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient


def _start_redis(port: int):
    if shutil.which("redis-server") is None:
        pytest.skip("redis-server not available")
    proc = subprocess.Popen([
        "redis-server",
        "--port",
        str(port),
        "--save",
        "",
        "--appendonly",
        "no",
    ])
    # Wait for server to start
    time.sleep(0.5)
    return proc


@pytest.fixture(scope="module")
def redis_url():
    port = 6380
    proc = _start_redis(port)
    url = f"redis://localhost:{port}/0"
    yield url
    proc.terminate()
    proc.wait()


def test_redis_rate_limiter(redis_url, monkeypatch):
    monkeypatch.setenv("NEURO_REDIS_URL", redis_url)
    monkeypatch.setenv("NEURO_API_KEY", "secret")
    monkeypatch.setenv("NEURO_RATE_LIMIT", "1")
    monkeypatch.setenv("NEURO_RATE_PERIOD", "60")

    spec = importlib.util.spec_from_file_location(
        "neurosales.security",
        str(resolve_path("neurosales/security.py")),
    )
    security = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(security)
    assert security.RateLimiter is security.RedisRateLimiter

    rl = security.RateLimiter()

    app = FastAPI()

    @app.get("/")
    async def root(api_key: str = Depends(security.get_api_key), _: None = Depends(rl)):
        return {"ok": True}

    client = TestClient(app)
    headers = {"X-API-Key": "secret"}
    client.get("/", headers=headers)
    r = client.get("/", headers=headers)
    assert r.status_code == 429


def test_archetype_cache_uses_env(redis_url, monkeypatch):
    monkeypatch.setenv("NEURO_REDIS_URL", redis_url)
    spec = importlib.util.spec_from_file_location(
        "neurosales.api_gateway",
        str(resolve_path("neurosales/api_gateway.py")),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = module.ArchetypeCache()
    cache.set("foo", "bar")
    assert cache.get("foo") == "bar"
