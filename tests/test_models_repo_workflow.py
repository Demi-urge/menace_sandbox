# flake8: noqa
import subprocess
import types
import threading
import time
from pathlib import Path
import os
import shutil
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import sys  # noqa: E402

sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    types.ModuleType("ed25519"),
)
serialization_mod = types.ModuleType("serialization")
sys.modules.setdefault(
    "cryptography.hazmat.primitives.serialization",
    serialization_mod,
)
sys.modules["cryptography.hazmat.primitives"].serialization = serialization_mod
sys.modules.setdefault(
    "yaml", types.SimpleNamespace(safe_dump=lambda data: "", safe_load=lambda d: {})
)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("celery", types.ModuleType("celery"))
sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.modules.setdefault("jinja2", types.SimpleNamespace(Template=lambda *a, **k: None))
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
sys.modules.setdefault("db_router", types.SimpleNamespace(DBRouter=object, DBResult=object))

vec_stub = types.ModuleType("vector_service")


class _CB:
    def __init__(self, *a, **k):
        pass

    def build(self, *_a, **_k):
        return ""


vec_stub.ContextBuilder = _CB
vec_stub.FallbackResult = type("FallbackResult", (), {})
vec_stub.ErrorResult = type("ErrorResult", (), {})
vec_stub.EmbeddableDBMixin = object
sys.modules.setdefault("vector_service", vec_stub)

import menace.implementation_pipeline as ip  # noqa: E402
import menace.bot_development_bot as bdb  # noqa: E402
from menace.bot_development_bot import BotSpec  # noqa: E402
import menace.task_handoff_bot as thb  # noqa: E402
import menace.models_repo as mrepo  # noqa: E402
from menace.models_repo import ACTIVE_MODEL_FILE  # noqa: E402



def _ctx_builder():
    return bdb.ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True)
    (path / "README.md").write_text("x")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True)


def test_clone_after_completion(tmp_path, monkeypatch):
    repo = tmp_path / "models"
    repo.mkdir()
    _init_repo(repo)

    monkeypatch.setenv("MODELS_REPO_PATH", str(repo))
    monkeypatch.setenv("MODELS_REPO_URL", repo.as_uri())
    monkeypatch.setattr(mrepo, "MODELS_REPO_PATH", repo)
    monkeypatch.setattr(mrepo, "MODELS_REPO_URL", repo.as_uri())
    monkeypatch.setattr(mrepo, "ACTIVE_MODEL_FILE", repo / ".active_model")
    monkeypatch.setattr(ip, "ACTIVE_MODEL_FILE", repo / ".active_model")
    monkeypatch.setattr(
        sys.modules[__name__], "ACTIVE_MODEL_FILE", repo / ".active_model"
    )
    monkeypatch.setattr(bdb, "Repo", None)
    monkeypatch.setattr(bdb, "ACTIVE_MODEL_FILE", repo / ".active_model")

    def fake_run(cmd, **kw):
        if cmd[:2] == ["git", "clone"] and len(cmd) >= 4:
            src, dest = Path(cmd[2]), Path(cmd[3])
            shutil.copytree(src, dest, dirs_exist_ok=True)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(ip.subprocess, "run", fake_run)
    monkeypatch.setattr(ip.subprocess, "run", fake_run)
    monkeypatch.setattr(ip.subprocess, "run", fake_run)

    builder = _ctx_builder()
    dev = bdb.BotDevelopmentBot(repo_base=repo, context_builder=builder)
    pipeline = ip.ImplementationPipeline(builder, developer=dev)
    tasks = [
        thb.TaskInfo(
            name="Bot",
            dependencies=[],
            resources={},
            schedule="once",
            code="",
            metadata={"purpose": "demo", "functions": ["run"]},
        )
    ]
    pipeline.run(tasks, model_id=1)
    assert (repo.parent / "1").exists()


def test_edit_waits_for_active(tmp_path, monkeypatch):
    repo = tmp_path / "models"
    repo.mkdir()
    _init_repo(repo)

    monkeypatch.setenv("MODELS_REPO_PATH", str(repo))
    monkeypatch.setenv("MODELS_REPO_URL", repo.as_uri())
    monkeypatch.setattr(mrepo, "MODELS_REPO_PATH", repo)
    monkeypatch.setattr(mrepo, "MODELS_REPO_URL", repo.as_uri())
    monkeypatch.setattr(mrepo, "ACTIVE_MODEL_FILE", repo / ".active_model")
    monkeypatch.setattr(ip, "ACTIVE_MODEL_FILE", repo / ".active_model")
    monkeypatch.setattr(
        sys.modules[__name__], "ACTIVE_MODEL_FILE", repo / ".active_model"
    )
    monkeypatch.setattr(bdb, "Repo", None)

    # Prepare source repository for cloning
    src_repo = repo.parent / "1"
    src_repo.mkdir()
    _init_repo(src_repo)

    def fake_run(cmd, **kw):
        if cmd[:2] == ["git", "clone"] and len(cmd) >= 4:
            src, dest = Path(cmd[2]), Path(cmd[3])
            shutil.copytree(src, dest, dirs_exist_ok=True)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(ip.subprocess, "run", fake_run)

    # Existing bot directory without git
    bot_dir = repo / "Bot"
    bot_dir.mkdir()

    dev = bdb.BotDevelopmentBot(repo_base=repo, context_builder=_ctx_builder())
    spec = BotSpec(name="Bot", purpose="demo", functions=["run"])

    ACTIVE_MODEL_FILE.write_text("busy")

    results: list[Path] = []

    def run_env():
        results.append(dev.create_env(spec, model_id=1))

    t = threading.Thread(target=run_env)
    start = time.time()
    t.start()
    time.sleep(0.2)
    ACTIVE_MODEL_FILE.unlink()
    t.join(timeout=5)
    elapsed = time.time() - start
    assert results and elapsed >= 0.2

    path = results[0]
    assert (path / ".git").exists()
    assert any(path.iterdir())


def test_clone_to_new_repo_pushes(tmp_path, monkeypatch):
    repo = tmp_path / "models"
    repo.mkdir()
    _init_repo(repo)

    push_base = tmp_path / "remote"

    monkeypatch.setenv("MODELS_REPO_PATH", str(repo))
    monkeypatch.setenv("MODELS_REPO_URL", repo.as_uri())
    monkeypatch.setenv("MODELS_REPO_PUSH_URL", push_base.as_uri())
    monkeypatch.setattr(mrepo, "MODELS_REPO_PATH", repo)
    monkeypatch.setattr(mrepo, "MODELS_REPO_URL", repo.as_uri())
    monkeypatch.setattr(mrepo, "MODELS_REPO_PUSH_URL", push_base.as_uri())

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["git", "clone"] and len(cmd) >= 4:
            src, dest = Path(cmd[2]), Path(cmd[3])
            shutil.copytree(src, dest, dirs_exist_ok=True)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    dest = mrepo.clone_to_new_repo(2)
    assert dest.exists()
    expected_remote = f"{push_base.as_uri().rstrip('/')}/2"
    assert any(c[:4] == ["git", "remote", "set-url", "origin"] and c[4] == expected_remote for c in calls)  # noqa: E501
    assert any(c[:2] == ["git", "push"] for c in calls)


def test_repeated_edits_refresh_directory(tmp_path, monkeypatch):
    repo = tmp_path / "models"
    repo.mkdir()
    _init_repo(repo)

    monkeypatch.setenv("MODELS_REPO_PATH", str(repo))
    monkeypatch.setenv("MODELS_REPO_URL", repo.as_uri())
    monkeypatch.setattr(mrepo, "MODELS_REPO_PATH", repo)
    monkeypatch.setattr(mrepo, "MODELS_REPO_URL", repo.as_uri())
    monkeypatch.setattr(bdb, "Repo", None)

    # source repo for model 1
    src1 = repo.parent / "1"
    src1.mkdir()
    _init_repo(src1)
    (src1 / "content.txt").write_text("one")
    subprocess.run(["git", "add", "content.txt"], cwd=src1, check=True)
    subprocess.run(["git", "commit", "-m", "one"], cwd=src1, check=True)

    # existing bot directory cloned from model 1
    bot_dir = repo / "Bot"
    shutil.copytree(src1, bot_dir)

    def fake_run(cmd, cwd=None, **kw):
        if cmd[:2] == ["git", "clone"] and len(cmd) >= 4:
            src, dest = Path(cmd[2]), Path(cmd[3])
            shutil.copytree(src, dest, dirs_exist_ok=True)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:2] == ["git", "fetch"] and len(cmd) >= 3:
            src = Path(cmd[2])
            dest = Path(cwd)
            shutil.rmtree(dest)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["git", "reset", "--hard"]:
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    dev = bdb.BotDevelopmentBot(repo_base=repo, context_builder=_ctx_builder())
    spec = BotSpec(name="Bot", purpose="demo", functions=["run"])
    path1 = dev.create_env(spec, model_id=1)
    assert (path1 / "content.txt").read_text() == "one"

    # source repo for model 2 with updated content
    src2 = repo.parent / "2"
    src2.mkdir()
    _init_repo(src2)
    (src2 / "content.txt").write_text("two")
    subprocess.run(["git", "add", "content.txt"], cwd=src2, check=True)
    subprocess.run(["git", "commit", "-m", "two"], cwd=src2, check=True)

    path2 = dev.create_env(spec, model_id=2)
    assert path2 == path1
    assert (path2 / "content.txt").read_text() == "two"


def test_model_build_lock_cleans_up(tmp_path, monkeypatch):
    repo = tmp_path / "models"
    repo.mkdir()

    monkeypatch.setattr(mrepo, "MODELS_REPO_PATH", repo)
    monkeypatch.setattr(mrepo, "ACTIVE_MODEL_FILE", repo / ".active_model")

    with pytest.raises(RuntimeError):
        with mrepo.model_build_lock(1):
            assert mrepo.ACTIVE_MODEL_FILE.exists()
            raise RuntimeError("boom")

    assert not mrepo.ACTIVE_MODEL_FILE.exists()


def test_pipeline_lock_removed_on_failure(tmp_path, monkeypatch):
    repo = tmp_path / "models"
    repo.mkdir()
    _init_repo(repo)

    monkeypatch.setenv("MODELS_REPO_PATH", str(repo))
    monkeypatch.setenv("MODELS_REPO_URL", repo.as_uri())
    monkeypatch.setattr(mrepo, "MODELS_REPO_PATH", repo)
    monkeypatch.setattr(mrepo, "MODELS_REPO_URL", repo.as_uri())
    monkeypatch.setattr(mrepo, "ACTIVE_MODEL_FILE", repo / ".active_model")
    monkeypatch.setattr(ip, "model_build_lock", mrepo.model_build_lock)
    monkeypatch.setattr(ip, "clone_to_new_repo", lambda mid: repo)
    monkeypatch.setattr(bdb, "Repo", None)
    monkeypatch.setattr(
        sys.modules[__name__], "ACTIVE_MODEL_FILE", repo / ".active_model"
    )
    monkeypatch.setattr(bdb, "ACTIVE_MODEL_FILE", repo / ".active_model")

    class FailingDev(bdb.BotDevelopmentBot):
        def build_from_plan(self, data: str, model_id=None):  # type: ignore[override]
            raise RuntimeError("fail")

    builder = _ctx_builder()
    pipeline = ip.ImplementationPipeline(
        builder,
        developer=FailingDev(repo_base=repo, context_builder=builder),
    )  # noqa: E501

    tasks = [
        thb.TaskInfo(
            name="Bot",
            dependencies=[],
            resources={},
            schedule="once",
            code="",
            metadata={"purpose": "demo", "functions": ["run"]},
        )
    ]

    with pytest.raises(RuntimeError):
        pipeline.run(tasks, model_id=1)

    assert not mrepo.ACTIVE_MODEL_FILE.exists()


def test_pipeline_waits_for_existing_marker(tmp_path, monkeypatch):
    repo = tmp_path / "models"
    repo.mkdir()
    _init_repo(repo)

    monkeypatch.setenv("MODELS_REPO_PATH", str(repo))
    monkeypatch.setenv("MODELS_REPO_URL", repo.as_uri())
    monkeypatch.setattr(mrepo, "MODELS_REPO_PATH", repo)
    monkeypatch.setattr(mrepo, "MODELS_REPO_URL", repo.as_uri())
    monkeypatch.setattr(mrepo, "ACTIVE_MODEL_FILE", repo / ".active_model")
    monkeypatch.setattr(ip, "model_build_lock", mrepo.model_build_lock)
    monkeypatch.setattr(ip, "clone_to_new_repo", lambda mid: repo)
    monkeypatch.setattr(bdb, "Repo", None)
    monkeypatch.setattr(
        sys.modules[__name__], "ACTIVE_MODEL_FILE", repo / ".active_model"
    )
    monkeypatch.setattr(bdb, "ACTIVE_MODEL_FILE", repo / ".active_model")

    class FailingDev(bdb.BotDevelopmentBot):
        def build_from_plan(self, data: str, model_id=None):  # type: ignore[override]
            raise RuntimeError("fail")

    builder = _ctx_builder()
    pipeline = ip.ImplementationPipeline(
        builder,
        developer=FailingDev(repo_base=repo, context_builder=builder),
    )  # noqa: E501

    tasks = [
        thb.TaskInfo(
            name="Bot",
            dependencies=[],
            resources={},
            schedule="once",
            code="",
            metadata={"purpose": "demo", "functions": ["run"]},
        )
    ]

    # Pre-create marker so the pipeline has to wait
    ACTIVE_MODEL_FILE.write_text("busy")

    errors: list[Exception] = []

    def run_pipeline():
        try:
            pipeline.run(tasks, model_id=1)
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=run_pipeline)
    start = time.time()
    t.start()
    time.sleep(0.2)
    ACTIVE_MODEL_FILE.unlink()
    t.join(timeout=5)
    elapsed = time.time() - start

    assert errors and isinstance(errors[0], RuntimeError)
    assert elapsed >= 0.2
    assert not mrepo.ACTIVE_MODEL_FILE.exists()
