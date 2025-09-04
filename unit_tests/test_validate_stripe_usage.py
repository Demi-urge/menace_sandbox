import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from codex_output_analyzer import CriticalGenerationFailure, validate_stripe_usage  # noqa: E402


def test_router_import_without_usage_raises() -> None:
    code = """
import stripe_billing_router


def create_invoice():
    invoice_id = "123"
    return invoice_id
"""
    with pytest.raises(CriticalGenerationFailure):
        validate_stripe_usage(code)


def test_router_import_with_call_passes() -> None:
    code = """
import stripe_billing_router


def process_payment():
    return stripe_billing_router.process_payment()
"""
    validate_stripe_usage(code)
