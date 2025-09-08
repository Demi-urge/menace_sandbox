from vector_service import ContextBuilder


def create_context_builder() -> ContextBuilder:
    """Return a :class:`ContextBuilder` wired to the standard local databases."""
    return ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
