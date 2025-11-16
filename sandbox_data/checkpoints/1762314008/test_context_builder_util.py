from pathlib import Path
from types import SimpleNamespace

import context_builder_util


class _StubContextBuilder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build(self, *args, **kwargs):
        return {"context": [], "metadata": {}}


def test_create_context_builder_bootstraps_missing_dbs(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    created_paths: list[Path] = []

    def _ensure_readable(path: Path, filename: str) -> str:
        created_paths.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        return str(path)

    stub_module = SimpleNamespace(
        _ensure_readable=_ensure_readable,
        ContextBuilder=_StubContextBuilder,
    )

    monkeypatch.setattr(context_builder_util, "_MODULE_CACHE", stub_module, raising=False)

    builder = context_builder_util.create_context_builder()

    expected = {"bots.db", "code.db", "errors.db", "workflows.db"}
    touched = {path.name for path in created_paths}
    assert expected.issubset(touched)

    for filename in expected:
        db_path = data_dir / filename
        assert db_path.exists() and db_path.is_file()
        assert str(db_path) == builder.kwargs[filename[:-3] + "_db"]

    monkeypatch.delenv("SANDBOX_DATA_DIR", raising=False)
