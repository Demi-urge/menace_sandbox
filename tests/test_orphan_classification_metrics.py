from pathlib import Path

import orphan_analyzer
from sandbox_runner.orphan_discovery import (
    discover_recursive_orphans,
    load_orphan_cache,
)


def _write_non_trivial(path: Path) -> None:
    path.write_text(
        '"""doc"""\n\n'
        'def foo(x):\n'
        '    print(x)\n'
        '    if x:\n'
        '        return 1\n'
        '    return 0\n'
    )


def test_trivial_isolated_module_redundant(tmp_path):
    mod = tmp_path / "trivial.py"
    mod.write_text("def foo():\n    pass\n")
    cls, meta = orphan_analyzer.classify_module(mod, include_meta=True)
    assert cls == "redundant"
    assert meta["calls"] == 0


def test_non_trivial_module_candidate(tmp_path):
    mod = tmp_path / "mod.py"
    _write_non_trivial(mod)
    cls, meta = orphan_analyzer.classify_module(mod, include_meta=True)
    assert cls == "candidate"
    assert meta["functions"] == 1
    assert meta["docstring"] is True
    assert meta["calls"] == 1


def test_orphan_discovery_records_metrics(tmp_path):
    mod = tmp_path / "mod.py"
    _write_non_trivial(mod)
    mapping = discover_recursive_orphans(str(tmp_path))
    info = mapping["mod"]
    assert info["classification"] == "candidate"
    assert info["functions"] == 1
    assert "complexity" in info
    assert info["calls"] == 1
    cache = load_orphan_cache(tmp_path)
    entry = cache[next(iter(cache))]
    assert entry["functions"] == 1
    assert entry["docstring"] is True
    assert entry["calls"] == 1


def test_custom_classifier_override(tmp_path):
    mod = tmp_path / "foo.py"
    mod.write_text("pass\n")

    def force_legacy(path: Path, metrics: dict) -> str:
        return "legacy"

    cls = orphan_analyzer.classify_module(mod, classifiers=[force_legacy])
    assert cls == "legacy"
