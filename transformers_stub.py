"""Legacy lightweight placeholder for optional Hugging Face transformers dependency.

This stub is retained for tests that manually insert it into ``sys.modules`` to
avoid importing the heavy :mod:`transformers` package.  The real library should
always be preferred when available, so keep this module name distinct from the
actual dependency to prevent accidental shadowing.
"""

AutoTokenizer = None
