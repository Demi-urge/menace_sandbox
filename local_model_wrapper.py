from __future__ import annotations

"""Wrapper around local Transformer models enforcing ContextBuilder usage."""

from typing import Any, Sequence

from snippet_compressor import compress_snippets

try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder, FallbackResult
except Exception:  # pragma: no cover - fallback stubs
    ContextBuilder = Any  # type: ignore

    class FallbackResult(list):  # type: ignore[misc]
        pass

try:  # pragma: no cover - optional dependency
    from vector_service import ErrorResult
except Exception:  # pragma: no cover - fallback stub
    class ErrorResult(Exception):  # type: ignore[override]
        pass


class LocalModelWrapper:
    """Minimal helper adding context retrieval before generation."""

    def __init__(self, hf_model: Any, tokenizer: Any) -> None:
        self.hf_model = hf_model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        *,
        context_builder: ContextBuilder,
        cb_input: str | None = None,
        **gen_kwargs: Any,
    ) -> str | list[str]:
        """Generate text with contextual snippets prepended.

        ``context_builder`` must be supplied as a keyword argument.  ``cb_input``
        allows providing a different query for retrieval; when omitted the
        ``prompt`` itself is used.
        """

        ctx_text = ""
        try:  # pragma: no cover - best effort context retrieval
            ctx_res = context_builder.build(cb_input or prompt)
            ctx_text = ctx_res[0] if isinstance(ctx_res, tuple) else ctx_res
            if isinstance(ctx_text, (FallbackResult, ErrorResult)):
                ctx_text = ""
            elif ctx_text:
                ctx_text = compress_snippets({"snippet": ctx_text}).get(
                    "snippet", ctx_text
                )
        except Exception:
            ctx_text = ""

        final_prompt = f"{ctx_text}\n\n{prompt}" if ctx_text else prompt
        input_ids = self.tokenizer.encode(final_prompt, return_tensors="pt")
        outputs: Sequence[Any] = self.hf_model.generate(input_ids, **gen_kwargs)
        decoded = [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

        def _strip(text: str) -> str:
            return text[len(final_prompt):] if text.startswith(final_prompt) else text

        decoded = [_strip(d) for d in decoded]
        return decoded[0] if len(decoded) == 1 else list(decoded)


__all__ = ["LocalModelWrapper"]
