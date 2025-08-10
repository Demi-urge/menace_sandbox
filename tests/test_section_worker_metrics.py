import asyncio
import types
import sys
import sandbox_runner.environment as env


def test_section_worker_returns_metrics(monkeypatch):
    monkeypatch.setattr(env, "_rlimits_supported", lambda: False)
    monkeypatch.setattr(env, "psutil", None)

    async def fake_exec(code, env_input, **kw):
        return {
            "exit_code": 0.0,
            "cpu": 1.0,
            "memory": 2.0,
            "disk_io": 3.0,
            "net_io": 4.0,
            "gpu_usage": 5.0,
        }

    monkeypatch.setattr(env, "_execute_in_container", fake_exec)
    monkeypatch.setattr(env.time, "sleep", lambda s: None)

    res, updates = asyncio.run(env._section_worker("print('x')", {}, 0.0))
    assert res["exit_code"] == 0.0
    assert updates
    metrics = updates[0][2]
    for key in ("cpu", "memory", "disk_io", "net_io", "gpu_usage"):
        assert key in metrics
        assert isinstance(metrics[key], float)
    assert "success_rate" in metrics
    assert isinstance(metrics["success_rate"], float)
    assert "avg_completion_time" in metrics
    assert isinstance(metrics["avg_completion_time"], float)


def test_section_worker_concurrency(monkeypatch):
    monkeypatch.setattr(env, "_rlimits_supported", lambda: False)
    monkeypatch.setattr(env, "psutil", None)

    async def fake_exec(code, env_input, **kw):
        return {"exit_code": 0.0, "cpu": 1.0}

    monkeypatch.setattr(env, "_execute_in_container", fake_exec)
    monkeypatch.setattr(env.time, "sleep", lambda s: None)

    res, updates = asyncio.run(
        env._section_worker("print('x')", {"CONCURRENCY_LEVEL": "2"}, 0.0)
    )
    assert res["exit_code"] == 0
    metrics = updates[0][2]
    assert metrics["concurrency_level"] == 2.0
    assert "concurrency_throughput" in metrics
    assert "concurrency_error_rate" in metrics
    assert metrics["success_rate"] == 1.0
    assert "avg_completion_time" in metrics
