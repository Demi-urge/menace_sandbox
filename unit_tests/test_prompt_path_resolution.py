import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.modules.pop("dynamic_path_router", None)
import dynamic_path_router as dpr  # noqa: E402
resolve_path = dpr.resolve_path
path_for_prompt = dpr.path_for_prompt
clear_cache = dpr.clear_cache


def _error_logger_prompt(module: str, hint: str) -> str:
    resolved = str(resolve_path(module))
    return f"Fix bottleneck in {resolved}: {hint}"


def test_error_logger_prompt_resolves_paths():
    prompt = _error_logger_prompt(path_for_prompt("enhancement_bot.py"), "hint")
    assert str(resolve_path("enhancement_bot.py")) in prompt


def test_prompt_resolves_after_repo_move(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "error_logger.py").write_text("pass", encoding="utf-8")  # path-ignore
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    clear_cache()
    prompt = _error_logger_prompt(path_for_prompt("error_logger.py"), "hint")
    assert str(repo / "error_logger.py") in prompt  # path-ignore
