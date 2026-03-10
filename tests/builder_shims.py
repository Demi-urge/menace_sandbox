"""Deterministic context-builder test shims."""


class BuildContextBuilderShim:
    """Tiny builder shim with explicit interface for tests."""

    def __init__(self, result: str = "") -> None:
        self._result = result

    def build(self, *_args, **_kwargs) -> str:
        return self._result

    def __getattr__(self, name: str):
        raise NotImplementedError(
            f"BuildContextBuilderShim does not support '{name}'. Supported methods: build."
        )
