from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import cast
import ast


def _load_iter_bot_modules() -> Callable[[Path], list[Path]]:
    """Return the ``_iter_bot_modules`` function from ``prime_registry.py``."""

    source_path = Path("prime_registry.py")
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)

    function_node = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_iter_bot_modules"
    )
    function_source = ast.get_source_segment(source, function_node)

    namespace: dict[str, object] = {}
    exec("from __future__ import annotations\nfrom pathlib import Path\n" + function_source, namespace)
    return cast(Callable[[Path], list[Path]], namespace["_iter_bot_modules"])


def test_iter_bot_modules_excludes_hidden_and_virtualenv(tmp_path: Path) -> None:
    """Hidden and virtual environment directories should be ignored."""

    iter_bot_modules = _load_iter_bot_modules()

    visible = tmp_path / "bots" / "visible_bot.py"
    hidden = tmp_path / ".git" / "hooks" / "ignored_bot.py"
    virtualenv = tmp_path / "venv" / "pkg" / "also_ignored_bot.py"
    dotted_dir = tmp_path / "src" / ".hidden" / "ignored_bot.py"

    for path in (visible, hidden, virtualenv, dotted_dir):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# dummy\n", encoding="utf-8")

    discovered = {path.relative_to(tmp_path) for path in iter_bot_modules(tmp_path)}

    assert Path("bots/visible_bot.py") in discovered
    assert all(
        excluded not in discovered
        for excluded in {
            Path(".git/hooks/ignored_bot.py"),
            Path("venv/pkg/also_ignored_bot.py"),
            Path("src/.hidden/ignored_bot.py"),
        }
    )
