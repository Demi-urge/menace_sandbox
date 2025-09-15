from __future__ import annotations

"""Wrapper around local Transformer models enforcing ContextBuilder usage."""

from typing import Any, Sequence

from snippet_compressor import compress_snippets  # noqa: F401 - retained for tests

try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - fallback stubs
    ContextBuilder = Any  # type: ignore

from prompt_types import Prompt


class LocalModelWrapper:
    """Minimal helper adding context retrieval before generation."""

    def __init__(self, hf_model: Any, tokenizer: Any) -> None:
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.last_prompt: Prompt | None = None

    def generate(
        self,
        prompt: Prompt | str,
        *,
        context_builder: ContextBuilder,
        cb_input: str | None = None,
        **gen_kwargs: Any,
    ) -> str | list[str]:
        """Generate text using a pre-built :class:`Prompt`.

        ``context_builder`` must be supplied as a keyword argument.  When
        ``prompt`` is provided as a plain string it is first converted into a
        :class:`Prompt` via :meth:`ContextBuilder.build_prompt`.  ``cb_input``
        allows supplying a different retrieval query in that case; the final
        user text remains ``prompt``.
        """

        if not isinstance(prompt, Prompt):
            prompt_obj = context_builder.build_prompt(cb_input or prompt)
            if cb_input:
                prompt_obj.user = str(prompt)
        else:
            prompt_obj = prompt

        self.last_prompt = prompt_obj

        pieces: list[str] = []
        if getattr(prompt_obj, "system", None):
            pieces.append(prompt_obj.system)
        pieces.extend(prompt_obj.examples)
        pieces.append(prompt_obj.user)
        prompt_text = "\n\n".join(pieces)

        if (prompt_obj.metadata or getattr(prompt_obj, "tags", None)) and "metadata" not in gen_kwargs:
            meta = dict(getattr(prompt_obj, "metadata", {}) or {})
            if getattr(prompt_obj, "tags", None):
                meta.setdefault("tags", list(prompt_obj.tags))
            gen_kwargs["metadata"] = meta

        input_ids = self.tokenizer.encode(prompt_text, return_tensors="pt")
        try:
            prompt_len = input_ids.shape[-1]
        except Exception:  # pragma: no cover - generic tensor/seq handling
            prompt_len = len(input_ids[0]) if input_ids else 0

        outputs: Sequence[Any] = self.hf_model.generate(input_ids, **gen_kwargs)
        decoded = [
            self.tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in outputs
        ]
        return decoded[0] if len(decoded) == 1 else list(decoded)


__all__ = ["LocalModelWrapper"]
