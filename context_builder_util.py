from importlib.machinery import SourceFileLoader
from pathlib import Path

_create_module = SourceFileLoader(
    "config.create_context_builder",
    str(Path(__file__).resolve().parent / "config" / "create_context_builder.py"),
).load_module()

create_context_builder = _create_module.create_context_builder
