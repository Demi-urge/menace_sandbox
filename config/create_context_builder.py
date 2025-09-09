try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - fallback for tests
    class ContextBuilder:  # type: ignore[misc]
        def __init__(self, *_, **__):  # pragma: no cover - simple stub
            pass


def create_context_builder() -> ContextBuilder:
    """Return a :class:`ContextBuilder` wired to the standard local databases."""
    try:
        return ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    except TypeError as exc:  # pragma: no cover - for simple stubs in tests
        raise ValueError(
            "ContextBuilder requires paths to 'bots.db', 'code.db', 'errors.db', "
            "and 'workflows.db'"
        ) from exc
