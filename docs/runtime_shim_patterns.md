# Runtime shim patterns

`python tools/qa/check_invalid_stubs.py` is a blocking QA check that prevents fragile runtime stubs.

`python tools/audit_placeholder_exports.py --enforce-baseline` is a Phase 1 guardrail that tracks module-level placeholder assignments (`None`, `object`, `SimpleNamespace`) and fails CI on **new** risky usages.

## Shim vs sentinel policy (Phase 1)

- **Shim**: a fallback object meant to be used like the real runtime dependency (called, or has methods/attributes accessed). Shims must be explicit classes or deterministic helpers.
- **Sentinel**: a private marker/handle used only for state signaling (for example `is None` checks, caching import errors, identity propagation), and **never** called or dereferenced.

If a symbol is ever called or has attributes touched, treat it as a shim, not a sentinel.

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
- Top-level `NAME = object` fallback sentinels used as service shims.
- Top-level `SimpleNamespace` service shims.
- Bare `pass` in executable branch fallbacks in shim-heavy modules.

## Placeholder-export audit + allowlist policy

The placeholder-export audit reports violations with file, symbol, assignment line, and dereference usage lines.

- `tools/qa/placeholder_audit_baseline.txt` stores existing debt so CI blocks only new findings.
- `ALLOWLIST_REASONS` in `tools/audit_placeholder_exports.py` is reserved for validated private sentinels and optional-dependency guarded internals that are never dereferenced.
- Allowlist entries must remain narrow (`(file, symbol)`), include a concrete reason, and should be removed once the symbol is migrated away from placeholder assignment.

Any broad file-wide or undocumented exception is not allowed.
