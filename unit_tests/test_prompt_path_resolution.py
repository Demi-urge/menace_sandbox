import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dynamic_path_router import resolve_path, path_for_prompt


def _visual_agent_prompt(path: str) -> str:
    resolved = str(resolve_path(path))
    return f"Path:{resolved}"


def _error_logger_prompt(module: str, hint: str) -> str:
    resolved = str(resolve_path(module))
    return f"Fix bottleneck in {resolved}: {hint}"


def test_visual_agent_prompt_resolves_paths():
    body = _visual_agent_prompt(path_for_prompt("error_logger.py"))
    assert str(resolve_path("error_logger.py")) in body


def test_error_logger_prompt_resolves_paths():
    prompt = _error_logger_prompt(path_for_prompt("enhancement_bot.py"), "hint")
    assert str(resolve_path("enhancement_bot.py")) in prompt
