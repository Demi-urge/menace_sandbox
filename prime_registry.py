from pathlib import Path
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

from menace_sandbox.bot_registry import BotRegistry
from menace_sandbox.bot_discovery import _iter_bot_modules

# Path to your persistent registry cache
persist_path = "bot_graph.db"

# Create or load registry
registry = BotRegistry(persist=persist_path)

# Discover and register all _bot.py files with their module paths
bot_dir = Path(__file__).resolve().parent
for module in _iter_bot_modules(bot_dir):
    print(f"Registering: {module.stem} -> {module}")
    registry.register_bot(module.stem, module_path=str(module))

# Save the populated registry
registry.save(persist_path)
print(f"Registry saved to {persist_path}")
