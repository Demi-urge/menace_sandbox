import pytest
from menace.codex_output_analyzer import (
    validate_stripe_usage,
    CriticalGenerationFailure,
)


def test_detect_import_stripe():
    code = "import stripe\n"
    with pytest.raises(CriticalGenerationFailure):
        validate_stripe_usage(code)


def test_detect_direct_api_call():
    code = "import requests\nrequests.get('https://api.stripe.com/v1')\n"
    with pytest.raises(CriticalGenerationFailure):
        validate_stripe_usage(code)


def test_detect_payment_keyword_without_router():
    code = "def checkout():\n    pass\n"
    with pytest.raises(CriticalGenerationFailure):
        validate_stripe_usage(code)


def test_allow_payment_with_router():
    code = (
        "import stripe_billing_router\n\n"
        "def payment():\n    return stripe_billing_router.charge()\n"
    )
    validate_stripe_usage(code)
