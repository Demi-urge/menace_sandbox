"""Legacy lightweight placeholder for optional Hugging Face transformers dependency.

This stub is retained for tests that manually insert it into ``sys.modules`` to
avoid importing the heavy :mod:`transformers` package.  The real library should
always be preferred when available, so keep this module name distinct from the
actual dependency to prevent accidental shadowing.

Temporary compatibility shim: :class:`AutoTokenizer` provides deterministic
callable behavior when the transformers dependency is absent.
"""


class AutoTokenizer:
    """Temporary compatibility shim for deterministic tokenizer behavior."""

    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get("model_name", "shim-tokenizer")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model_name = args[0] if args else kwargs.get("pretrained_model_name_or_path", "shim-tokenizer")
        return cls(model_name=model_name)

    def __call__(self, text, *args, **kwargs):
        text = "" if text is None else str(text)
        tokens = text.split()
        return {"input_ids": list(range(len(tokens))), "attention_mask": [1] * len(tokens)}
