from __future__ import annotations

from pathlib import Path
import ast
import logging
import sqlite3
from contextlib import closing
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

class _RegistryWriter:
    """Persist bot-module mappings to the on-disk registry."""

    def __init__(self) -> None:
        self._modules: dict[str, str] = {}

    def register(self, name: str, module_path: Path) -> None:
        self._modules[name] = str(module_path)

    def save(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(dest)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_nodes(
                    name TEXT PRIMARY KEY,
                    module TEXT,
                    version INTEGER,
                    last_good_module TEXT,
                    last_good_version INTEGER
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_edges(
                    from_bot TEXT,
                    to_bot TEXT,
                    weight REAL,
                    PRIMARY KEY(from_bot, to_bot)
                )
                """
            )
            cur.execute("DELETE FROM bot_nodes")
            cur.execute("DELETE FROM bot_edges")
            for name, module in sorted(self._modules.items()):
                cur.execute(
                    """
                    INSERT OR REPLACE INTO bot_nodes(
                        name, module, version, last_good_module, last_good_version
                    ) VALUES(?, ?, NULL, NULL, NULL)
                    """,
                    (name, module),
                )
            conn.commit()

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
        "sandbox_data",
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

        # ``Path.parts`` preserves the name of every directory component.  When
        # the repository is connected to a remote, Git stores remote tracking
        # branches under ``.git/refs`` using the branch name as the file name.
        # Some of our historical branches end with ``_bot.py`` which satisfies
        # the glob above.  Traversing those reference files caused the
        # registry priming command to attempt ``ast.parse`` on Git metadata,
        # resulting in noisy "invalid decimal literal" warnings and slowing the
        # boot process to a crawl.  Filtering out anything within a VCS
        # directory keeps the discovery focused on actual Python modules.
        if any(part in {".git", ".hg", ".svn"} for part in parts):
            continue

        # ``git`` worktrees may place the repository metadata outside of
        # ``root`` while exposing it through ``.git`` entries.  When those
        # entries contain branch names that happen to end in ``_bot.py`` (for
        # example ``fix-foo-bot.py``), ``rglob`` will happily return them and we
        # would later attempt to ``ast.parse`` their contents.  Git reference
        # files are not Python source files which results in noisy syntax
        # warnings and, on some systems, a noticeable slowdown while the parser
        # churns through the VCS metadata.  Filtering out anything under a VCS
        # directory keeps the discovery focused on actual source modules.
        # ``sandbox_data`` holds runtime checkpoints and other ephemeral
        # artefacts that can contain partially written Python files.  Skipping
        # the directory prevents ``ast.parse`` warnings when stale checkpoints
        # include content such as leading-zero literals.
        if any(part in ignore or part.startswith("test") for part in parts):
            continue
        if any(part.startswith(".") for part in parts):
            continue
        modules.append(path)
    return modules

def main() -> None:
    """Discover self-managed bots and persist the registry mapping."""

    # Path to your persistent registry cache
    persist_path = Path(__file__).resolve().with_name("bot_graph.db")

    # Collect discovered modules in a lightweight writer to avoid importing the
    # full runtime registry (which pulls in hundreds of supporting modules).
    registry = _RegistryWriter()

    # Discover and register all _bot.py files with their module paths
    bot_dir = Path(__file__).resolve().parent
    registered: dict[str, Path] = {}
    for module in _iter_bot_modules(bot_dir):
        for class_name in _iter_decorated_bot_classes(module):
            registry.register(class_name, module)
            registered[class_name] = module

    if registered:
        logger.info("Registered bot mappings:")
        for class_name, module in sorted(registered.items()):
            logger.info(" - %s -> %s", class_name, module.name)
    else:
        logger.info("No decorated bots discovered.")

    # Save the populated registry
    registry.save(persist_path)
    logger.info("Registry saved to %s", persist_path)


if __name__ == "__main__":
    main()
