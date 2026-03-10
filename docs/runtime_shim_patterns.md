# Runtime shim patterns

`python scripts/check_invalid_stubs.py` is a blocking QA check that prevents fragile runtime stubs.

## Accepted patterns

When an optional dependency is missing, use explicit shims that are safe and deterministic:

1. **Callable class shims (preferred)**
   - Provide a class with explicit methods (`__call__`, `run`, `listen`, etc.).
   - Keep method outputs deterministic for a given input.
   - Avoid hidden I/O or side effects.

2. **Deterministic method-only service shims**
   - Replace `SimpleNamespace(...)` placeholders with small classes.
   - Keep behavior obvious and testable (`return None`, raise explicit errors, or return fixed values).

3. **Explicit fallback control flow**
   - If a value starts as `None`, guard every call path (`if service is not None`) or raise a clear error before use.
   - Do not rely on implicit `None` callability.

## Disallowed patterns

- Top-level `NAME = None` that is later called without a proven guard.
- Top-level `NAME = object` fallback sentinels.
- Top-level `SimpleNamespace` service shims.
- Bare `pass` in executable branch fallbacks in shim-heavy modules.

## Allowlist policy

Any exception must be added in `scripts/check_invalid_stubs.py` with:

- exact file path,
- exact check code/symbol,
- a concrete explanation of why the fallback is safe.

Broad file-wide or undocumented exceptions are not allowed.
