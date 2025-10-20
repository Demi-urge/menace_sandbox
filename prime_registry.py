from pathlib import Path
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
