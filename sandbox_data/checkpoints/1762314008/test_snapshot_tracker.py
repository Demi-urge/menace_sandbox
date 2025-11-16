import types
import importlib
import json
import sys
from pathlib import Path

# Prepare minimal package structure and stubs
root_path = Path(__file__).resolve().parents[1]
pkg_path = root_path / "self_improvement"
root_pkg = types.ModuleType("menace_sandbox")
root_pkg.__path__ = [str(root_path)]
sys.modules.setdefault("menace_sandbox", root_pkg)
package = types.ModuleType("menace_sandbox.self_improvement")
package.__path__ = [str(pkg_path)]
sys.modules.setdefault("menace_sandbox.self_improvement", package)

# Stub dependencies required by snapshot_tracker
sys.modules.setdefault("audit_logger", types.SimpleNamespace(log_event=lambda *a, **k: None))
sys.modules.setdefault("menace_sandbox.audit_logger", sys.modules["audit_logger"])
sys.modules.setdefault(
    "snapshot_history_db",
    types.SimpleNamespace(
        log_regression=lambda *a, **k: None,
        record_snapshot=lambda *a, **k: 1,
        record_delta=lambda *a, **k: None,
    ),
)
sys.modules.setdefault(
    "menace_sandbox.snapshot_history_db", sys.modules["snapshot_history_db"]
)
sys.modules.setdefault("module_index_db", types.SimpleNamespace(ModuleIndexDB=None))
sys.modules.setdefault(
    "menace_sandbox.module_index_db", sys.modules["module_index_db"]
)
sys.modules.setdefault(
    "relevancy_radar", types.SimpleNamespace(call_graph_complexity=lambda files: 0)
)
sys.modules.setdefault(
    "menace_sandbox.relevancy_radar", sys.modules["relevancy_radar"]
)
sys.modules.setdefault(
    "dynamic_path_router",
    types.SimpleNamespace(resolve_path=lambda p: p, repo_root=lambda: Path(".")),
)

ss = importlib.import_module("menace_sandbox.self_improvement.snapshot_tracker")


def test_checkpoint_and_confidence(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "resolve_path", lambda p: p)

    repo = tmp_path / ss.resolve_path("repo")
    repo.mkdir()
    module = repo / ss.resolve_path("module.py")  # path-ignore
    module.write_text("a = 1\n", encoding="utf-8")

    data_dir = tmp_path / ss.resolve_path("data")

    class SettingsStub:
        sandbox_data_dir = str(data_dir)
        sandbox_repo_path = str(repo)
        snapshot_metrics = {"roi", "sandbox_score"}
        roi_drop_threshold = -1.0
        entropy_regression_threshold = 1e9

    monkeypatch.setattr(ss, "SandboxSettings", lambda: SettingsStub())

    tracker = ss.SnapshotTracker()
    tracker.capture(
        "before",
        {"files": [module], "roi": 1.0, "sandbox_score": 1.0},
        repo_path=repo,
    )

    diff = repo / ss.resolve_path("change.diff")
    diff.write_text(
        f"""diff --git a/{module.name} b/{module.name}
--- a/{module.name}
+++ b/{module.name}
@@
-a = 1
+a = 2
""",
        encoding="utf-8",
    )

    module.write_text("a = 2\n", encoding="utf-8")
    tracker.capture(
        "after",
        {
            "files": [module],
            "roi": 2.0,
            "sandbox_score": 2.0,
            "prompt": {"strategy": "alpha"},
            "diff": str(diff),
        },
        repo_path=repo,
    )
    tracker.delta()

    ckpt_base = data_dir / ss.resolve_path("checkpoints")
    dirs = list(ckpt_base.iterdir())
    assert len(dirs) == 1
    assert (dirs[0] / module.name).exists()

    conf = json.loads((data_dir / ss.resolve_path("strategy_confidence.json")).read_text())
    assert conf["alpha"] == 1


def test_capture_uses_repo_when_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "resolve_path", lambda p: p)

    repo = tmp_path / ss.resolve_path("repo")
    repo.mkdir()
    (repo / "a.py").write_text("a = 1\n", encoding="utf-8")  # path-ignore
    sub = repo / "sub"
    sub.mkdir()
    (sub / "b.py").write_text("b = 2\n", encoding="utf-8")  # path-ignore

    data_dir = tmp_path / ss.resolve_path("data")

    class SettingsStub:
        sandbox_data_dir = str(data_dir)
        sandbox_repo_path = str(repo)
        snapshot_metrics = {"entropy", "token_diversity"}
        roi_drop_threshold = -1.0
        entropy_regression_threshold = 1e9

    monkeypatch.setattr(ss, "SandboxSettings", lambda: SettingsStub())

    captured: dict[str, list[Path]] = {}

    def fake_collect(files, settings=None):
        flist = [Path(f) for f in files]
        captured["files"] = flist
        return float(len(flist)), float(len(flist))

    monkeypatch.setattr(ss, "collect_snapshot_metrics", fake_collect)

    snap = ss.capture("pre", [], roi=0.0, sandbox_score=0.0)

    assert snap.entropy == 2.0
    assert snap.token_diversity == 2.0
    assert set(captured["files"]) == {repo / "a.py", sub / "b.py"}  # path-ignore
