# Implementation Optimiser Bot

`ImplementationOptimiserBot` refines `TaskPackage` objects before they reach the developer stage. It keeps a history of processed packages and can infer missing implementations when only task metadata is supplied.

## `_optimise_python`

`_optimise_python(code: str)` performs a small set of AST transformations:

- removes `pass` statements when other statements exist in the same block
- strips blank module, class and function docstrings
- drops unused `import` and `from ... import` entries

If parsing fails the original text is returned unchanged.

## Code templates

When a task has no implementation, `fill_missing()` can supply one. Two Python styles are available:

- **minimal** – defines each required function and returns `True` or `False` depending on whether the body succeeds
- **logging** – wraps the body in `try`/`except`, emits `logger` messages and returns either a list of dependency results or a string value

Shell tasks receive a simple `#!/bin/sh` template with a `main` function that calls dependent tasks when present.

## Using `SelfCodingEngine`

If an instance of `SelfCodingEngine` is passed to the bot, `fill_missing()` first
calls `manager_generate_helper(manager, desc)` to produce an implementation based on task
metadata.  The engine retries with `CODEX_RETRY_DELAYS`, simplifies the prompt
on persistent failures and falls back to queueing or rerouting to
`CODEX_FALLBACK_MODEL` depending on `CODEX_FALLBACK_STRATEGY`.  Even when a
prompt is queued the bot proceeds with a stub so optimisation continues in
degraded mode.  When the engine is absent the templates above are used instead.

