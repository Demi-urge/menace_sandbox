"""Minimal sqlalchemy stub for sandbox runs without the dependency."""

from types import SimpleNamespace

from .engine import Engine, create_engine, make_url  # noqa: F401
from .exc import ArgumentError  # noqa: F401


class _Type:
    def __init__(self, *args, **kwargs) -> None:
        pass


Boolean = _Type
Float = _Type
Integer = _Type
String = _Type
Text = _Type


class ForeignKey:
    def __init__(self, target: str, *args, **kwargs) -> None:
        self.target = target


class Column:
    def __init__(self, name: str, *_args, **_kwargs) -> None:
        self.name = name


class MetaData:
    def create_all(self, _engine: Engine) -> None:
        return None


class Table:
    def __init__(self, name: str, metadata: MetaData, *columns, **_kwargs) -> None:
        self.name = name
        self.metadata = metadata
        self.columns = columns
        self.c = SimpleNamespace(
            **{col.name: col for col in columns if hasattr(col, "name")}
        )


class Index:
    def __init__(self, *_args, **_kwargs) -> None:
        pass


event = SimpleNamespace(listen=lambda *_args, **_kwargs: None)

__all__ = [
    "ArgumentError",
    "Boolean",
    "Column",
    "Engine",
    "Float",
    "ForeignKey",
    "Index",
    "Integer",
    "MetaData",
    "String",
    "Table",
    "Text",
    "create_engine",
    "event",
    "make_url",
]
