"""Deterministic runtime shims for optional dependencies."""

from __future__ import annotations


class UnifiedEventBusShim:
    """Minimal stand-in for ``UnifiedEventBus`` when the module is absent."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.persist_path = kwargs.get("persist_path")
        self._subscribers: dict[str, list[object]] = {}

    def publish(self, *args: object, **kwargs: object) -> None:
        return None

    def subscribe(self, topic: str, handler: object) -> None:
        self._subscribers.setdefault(topic, []).append(handler)


class SQLParseKeywordShim:
    """Identity marker matching ``sqlparse.tokens.Keyword`` checks."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.name = kwargs.get("name", "Keyword")


class EmptyVectorIndexShim:
    """Deterministic vector-index shim returning no hits."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.dim = kwargs.get("dim", 0)
        self.d = self.dim

    def search(self, _query: object, limit: int) -> tuple[list[float], list[list[int]]]:
        safe_limit = max(0, int(limit))
        return ([], [[-1] * safe_limit])

    def get_nns_by_vector(self, _embedding: object, _limit: int) -> list[int]:
        return []

    def add_with_ids(self, *_args: object, **_kwargs: object) -> None:
        return None


class ContextBuilderShim:
    """Placeholder context builder used by lightweight generation scripts."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.metadata = dict(kwargs)

    def refresh_db_weights(self) -> None:
        return None


class QuickFixShim:
    """Marker object indicating quick-fix bootstrap is intentionally deferred."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.enabled = bool(kwargs.get("enabled", False))

    def run(self, *args: object, **kwargs: object) -> None:
        return None
