import pathlib
import sys

import pytest
import base64

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from codex_output_analyzer import (  # noqa: E402
    CriticalGenerationFailure,
    validate_stripe_usage,
    validate_stripe_usage_generic,
)
from stripe_detection import contains_payment_keyword  # noqa: E402


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


def test_js_snippet_without_router_raises() -> None:
    code = "fetch('https://api.stripe.com/v1/charges')"
    with pytest.raises(CriticalGenerationFailure):
        validate_stripe_usage(code)


def test_js_snippet_with_router_import_passes() -> None:
    code = "import stripe_billing_router\nfetch('https://api.stripe.com/v1/charges')"
    validate_stripe_usage(code)


def test_plain_text_keyword_without_router_raises() -> None:
    text = "This page handles billing for users"
    with pytest.raises(CriticalGenerationFailure):
        validate_stripe_usage_generic(text)


def test_html_snippet_keyword_without_router_raises() -> None:
    html = "<div>Start checkout now</div>"
    with pytest.raises(CriticalGenerationFailure):
        validate_stripe_usage_generic(html)


def test_js_snippet_keyword_without_router_raises() -> None:
    js = "function sendInvoice() { return true; }"
    with pytest.raises(CriticalGenerationFailure):
        validate_stripe_usage_generic(js)


def test_camel_case_identifier_without_router_raises() -> None:
    code = """
def processPayment():
    return True
"""
    with pytest.raises(CriticalGenerationFailure):
        validate_stripe_usage(code)


def test_hyphenated_identifier_without_router_raises() -> None:
    js = "function send-Invoice() { return true; }"
    with pytest.raises(CriticalGenerationFailure):
        validate_stripe_usage_generic(js)


def test_contains_payment_keyword_camel_case() -> None:
    assert contains_payment_keyword("processPayment")


def test_contains_payment_keyword_hyphen() -> None:
    assert contains_payment_keyword("send-Invoice")


def test_generic_text_with_router_passes() -> None:
    text = "Payments are processed via stripe_billing_router"
    validate_stripe_usage_generic(text)


def test_base64_encoded_key_without_router_raises() -> None:
    encoded = base64.b64encode(b"sk_" + b"live_" + b"12345678").decode("ascii")
    with pytest.raises(CriticalGenerationFailure):
        validate_stripe_usage_generic(encoded)


def test_partially_masked_key_without_router_raises() -> None:
    masked = "sk_" + "live_" + "****1234"
    with pytest.raises(CriticalGenerationFailure):
        validate_stripe_usage_generic(masked)
