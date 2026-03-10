"""Minimal sqlalchemy stub for sandbox runs without the dependency."""

from types import SimpleNamespace

from .engine import Engine, create_engine, make_url  # noqa: F401
from .exc import ArgumentError  # noqa: F401


class _Type:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


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
        self.args = _args
        self.kwargs = _kwargs


class EventShim:
    """Deterministic shim for the tiny subset of sqlalchemy.event used here."""

    @staticmethod
    def listen(*_args, **_kwargs) -> None:
        return None

    def __getattr__(self, name: str):
        raise NotImplementedError(
            f"EventShim does not support '{name}'. Supported methods: listen."
        )


event = EventShim()

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
