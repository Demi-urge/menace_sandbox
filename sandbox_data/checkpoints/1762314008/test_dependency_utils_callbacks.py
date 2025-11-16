import logging

import pytest

from sandbox_runner.dependency_utils import collect_local_dependencies


def _make_repo(tmp_path):
    a = tmp_path / "a.py"  # path-ignore
    b = tmp_path / "b.py"  # path-ignore
    a.write_text("import b\n")
    b.write_text("x = 1\n")
    return a, b


def test_logs_on_module_error(tmp_path, monkeypatch, caplog):
    a, _ = _make_repo(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    def bad_module(rel, path, parents):  # pragma: no cover - explicit test
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        collect_local_dependencies([str(a)], on_module=bad_module)

    assert any("on_module callback failed" in r.message for r in caplog.records)


def test_logs_on_dependency_error(tmp_path, monkeypatch, caplog):
    a, _ = _make_repo(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    def bad_dep(dep_rel, parent_rel, chain):  # pragma: no cover - explicit test
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        collect_local_dependencies([str(a)], on_dependency=bad_dep)

    assert any("on_dependency callback failed" in r.message for r in caplog.records)


@pytest.mark.parametrize("cb_name", ["on_module", "on_dependency"])
def test_strict_propagates_exceptions(tmp_path, monkeypatch, cb_name):
    a, _ = _make_repo(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    def bad(*args, **kwargs):
        raise RuntimeError("fail")

    kwargs = {cb_name: bad, "strict": True}
    with pytest.raises(RuntimeError):
        collect_local_dependencies([str(a)], **kwargs)

