import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "dynamic_path_router", Path(__file__).resolve().parent.parent / "dynamic_path_router.py"
)
dr = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(dr)
resolve_path = dr.resolve_path


def _visual_agent_prompt(path: str) -> str:
    resolved = str(resolve_path(path))
    return f"Path:{resolved}"


def _error_logger_prompt(module: str, hint: str) -> str:
    resolved = str(resolve_path(module))
    return f"Fix bottleneck in {resolved}: {hint}"


def test_visual_agent_prompt_resolves_paths():
    body = _visual_agent_prompt("error_logger.py")
    assert str(resolve_path("error_logger.py")) in body


def test_error_logger_prompt_resolves_paths():
    prompt = _error_logger_prompt("enhancement_bot.py", "hint")
    assert str(resolve_path("enhancement_bot.py")) in prompt
