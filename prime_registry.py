from __future__ import annotations

from pathlib import Path
import ast
import logging
import sys


def _ensure_package_importable() -> None:
    """Ensure the project package can be imported when running as a script."""
    project_root = Path(__file__).resolve().parent
    parent_dir = project_root.parent
    # When this file is executed directly, Python only adds the directory of the
    # script (``project_root``) to ``sys.path``.  The package name
    # ``menace_sandbox`` resolves to that directory, so the parent directory must
    # also be on the import path.  This mirrors the behaviour of
    # ``python -m menace_sandbox.prime_registry`` but keeps the convenience of a
    # direct invocation.
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))


_ensure_package_importable()

from menace_sandbox import db_router
from menace_sandbox.bot_registry import BotRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _iter_decorated_bot_classes(module: Path) -> list[str]:
    """Yield class names decorated with ``self_coding_managed`` in *module*."""

    try:
        source = module.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Unable to read %s: %s", module, exc)
        return []

    try:
        tree = ast.parse(source, filename=str(module))
    except SyntaxError as exc:
        logger.warning("Failed parsing %s: %s", module, exc)
        return []

    def has_self_coding_managed(decorators: list[ast.expr]) -> bool:
        for decorator in decorators:
            target = decorator
            if isinstance(decorator, ast.Call):
                target = decorator.func
            if isinstance(target, ast.Attribute) and target.attr == "self_coding_managed":
                return True
            if isinstance(target, ast.Name) and target.id == "self_coding_managed":
                return True
        return False

    discovered: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not node.name.endswith("Bot"):
            continue
        if not has_self_coding_managed(node.decorator_list):
            continue
        discovered.append(node.name)
    return discovered


def _iter_bot_modules(root: Path) -> list[Path]:
    """Return bot module paths under *root* while skipping test helpers."""

    ignore = {
        "tests",
        "unit_tests",
        "docs",
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
    }
    modules: list[Path] = []
    for path in root.rglob("*_bot.py"):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            parts = path.parts

        if any(part in ignore or part.startswith("test") for part in parts):
            continue
        if any(part.startswith(".") for part in parts):
            continue
        modules.append(path)
    return modules

# Path to your persistent registry cache
persist_path = Path(__file__).resolve().with_name("bot_graph.db")

# Reset any pre-existing router so the generated database uses ``persist_path``.
db_router.GLOBAL_ROUTER = None

# Create or load registry
registry = BotRegistry()

# Discover and register all _bot.py files with their module paths
bot_dir = Path(__file__).resolve().parent
registered: dict[str, Path] = {}
for module in _iter_bot_modules(bot_dir):
    for class_name in _iter_decorated_bot_classes(module):
        registry.register_bot(class_name, module_path=str(module))
        registered[class_name] = module

if registered:
    logger.info("Registered bot mappings:")
    for class_name, module in sorted(registered.items()):
        logger.info(" - %s -> %s", class_name, module.name)
else:
    logger.info("No decorated bots discovered.")

# Save the populated registry
persist_path.parent.mkdir(parents=True, exist_ok=True)
registry.save(persist_path)
logger.info("Registry saved to %s", persist_path)
