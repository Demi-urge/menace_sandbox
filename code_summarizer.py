"""Lightweight helper to summarise code snippets.

The function tries multiple local approaches in order of preference:

1. :mod:`micro_models.diff_summarizer` – a fine tuned model specialised in
   code diffs.  The code is passed as the ``after`` portion of a synthetic diff
   with an empty ``before``.  The ``max_summary_tokens`` parameter is forwarded
   to the underlying model to cap the length of the generated summary.
2. :class:`llm_interface.LLMClient` – if a local LLM server compatible with the
   :mod:`local_client` helpers is available the code is summarised via a normal
   language model call.
3. Heuristic fallback – as a last resort the first non-empty line of the code
   is returned, truncated to ``max_summary_tokens`` token-like units.

All returned summaries are truncated to ``max_summary_tokens`` to avoid runaway
outputs when model limits are ignored or unavailable.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

from context_builder import PromptBuildError, handle_failure

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from llm_interface import LLMClient, Prompt  # noqa: F401
    from vector_service.context_builder import ContextBuilder  # noqa: F401


LOGGER = logging.getLogger(__name__)


def _emit_prompt_failure(message: str, exc: BaseException) -> None:
    """Log prompt construction errors in a consistent manner."""

    try:
        handle_failure(message, exc, logger=LOGGER)
    except PromptBuildError:
        # ``handle_failure`` always raises ``PromptBuildError``; swallow it so
        # the summariser can fall back to heuristic behaviour when possible.
        pass


def _build_summarisation_prompt(
    code: str, *, context_builder: "ContextBuilder"
) -> "Prompt":
    """Return an enriched prompt for the summarisation task."""

    from self_coding_engine import SelfCodingEngine

    engine = SelfCodingEngine.__new__(SelfCodingEngine)
    engine.logger = LOGGER
    engine._last_retry_trace = None
    engine._last_prompt = None
    engine._last_prompt_metadata = {}
    return engine.build_enriched_prompt(
        code,
        intent={"task": "summarize_code", "small_task": True},
        context_builder=context_builder,
    )


def _truncate_tokens(text: str, limit: int) -> str:
    """Return ``text`` truncated to at most ``limit`` whitespace tokens."""

    tokens = text.split()
    if len(tokens) <= limit:
        return text.strip()
    return " ".join(tokens[:limit]).strip()


def _heuristic_summary(code: str, limit: int) -> str:
    """Return a simple summary using AST/tokenisation heuristics."""

    try:
        import ast
        import io
        import tokenize

        tree = ast.parse(code)
        lines = code.splitlines()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                line = lines[node.lineno - 1].strip()
                return _truncate_tokens(line, limit)
    except Exception:
        try:
            for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                if tok.type == tokenize.NAME and tok.string in {"def", "class"}:
                    return _truncate_tokens(tok.line.strip(), limit)
        except Exception:
            pass

    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    if not lines:
        return ""
    return _truncate_tokens(lines[0], limit)


def summarize_code(
    code: str,
    *,
    context_builder: "ContextBuilder",
    max_summary_tokens: int = 128,
) -> str:
    """Return a short description of ``code``.

    The function attempts several local summarisation strategies before falling
    back to a simple heuristic based on the first line of the snippet.  When the
    micro-model path is unavailable the provided :class:`ContextBuilder` is used
    to build an enriched prompt with vector context and metadata.  If prompt
    generation fails, a simple heuristic is used as a last resort.
    """

    code = code.strip()
    if not code:
        return ""

    # ------------------------------------------------------------------
    # 1) Try the fine-tuned micro model
    try:  # pragma: no cover - optional dependency
        from micro_models.diff_summarizer import summarize_diff as _summ
    except Exception:  # pragma: no cover - summariser may be missing
        _summ = None
    if _summ is not None:
        try:  # pragma: no cover - defensive against runtime failures
            summary = _summ("", code, max_new_tokens=max_summary_tokens)
            if summary:
                return _truncate_tokens(summary, max_summary_tokens)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 2) Fallback to a local LLM client
    try:  # pragma: no cover - optional dependency
        from local_client import OllamaClient
    except Exception:  # pragma: no cover - client may be missing
        OllamaClient = None  # type: ignore[assignment]

    if OllamaClient is not None:
        try:  # pragma: no cover - defensive against runtime failures
            client = OllamaClient()  # type: ignore[call-arg]
        except Exception:
            pass
        else:
            try:
                prompt_obj = _build_summarisation_prompt(
                    code, context_builder=context_builder
                )
            except PromptBuildError as exc:
                _emit_prompt_failure(
                    "failed to build summarisation prompt", exc
                )
            except Exception as exc:
                _emit_prompt_failure(
                    "unexpected error while building summarisation prompt",
                    exc,
                )
            else:
                try:
                    result = client.generate(
                        prompt_obj, context_builder=context_builder
                    )
                    summary = getattr(result, "text", "").strip()
                except Exception as exc:
                    _emit_prompt_failure(
                        "local client failed to generate code summary", exc
                    )
                else:
                    if summary:
                        return _truncate_tokens(summary, max_summary_tokens)

    # ------------------------------------------------------------------
    # 3) Heuristic fallback
    return _heuristic_summary(code, max_summary_tokens)


__all__ = ["summarize_code"]
