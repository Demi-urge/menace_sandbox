import sys
import types
from pathlib import Path


def _load_func():
    path = Path(__file__).resolve().parents[1] / "run_autonomous.py"  # path-ignore
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("def _visual_agent_running"))
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines) and (
        not lines[end].strip() or lines[end].startswith(" " * (indent + 1))
    ):
        end += 1
    src = "\n".join(lines[start:end])
    ns = {
        "logger": types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            exception=lambda *a, **k: None,
        )
    }
    exec("import sys, types\n" + src, ns)
    return ns["_visual_agent_running"]


_visual_agent_running = _load_func()


def test_visual_agent_running_tries_multiple_urls(monkeypatch):
    calls = []

    def fake_get(url, timeout=0):
        calls.append((url, timeout))
        if "good" in url:
            class Resp:
                status_code = 200
            return Resp()
        raise RuntimeError("boom")

    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(get=fake_get),
    )

    assert _visual_agent_running("http://bad,1;http://good,2") is True
    assert calls == [
        ("http://bad/health", 1.0),
        ("http://good/health", 2.0),
    ]


def test_visual_agent_running_all_fail(monkeypatch):
    def fake_get(url, timeout=0):
        raise RuntimeError("boom")

    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(get=fake_get),
    )

    assert not _visual_agent_running("http://bad,1;http://worse,2")
