import ast
from pathlib import Path


def _imports(path: str) -> set[str]:
    tree = ast.parse(Path(path).read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_no_direct_imports_between_cycle_and_bot_creation():
    cycle_imports = _imports("sandbox_runner/cycle.py")  # path-ignore
    bot_imports = _imports("bot_creation_bot.py")  # path-ignore
    assert "bot_creation_bot" not in cycle_imports
    assert all(not name.startswith("bot_creation_bot") for name in cycle_imports)
    assert "sandbox_runner" not in bot_imports
    assert all(not name.startswith("sandbox_runner") for name in bot_imports)
