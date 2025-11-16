from secret_redactor import redact_secrets


def test_regex_redaction():
    text = "password=foobar AKIA1234567890ABCD12"
    redacted = redact_secrets(text)
    assert "foobar" not in redacted
    assert "AKIA1234567890ABCD12" not in redacted
    assert redacted.count("[REDACTED]") >= 2


def test_entropy_redaction():
    token = "ODktkNEfN5sdL1PpQZBRLXIanFkFJxuxMxDPZWS9VyQ"
    text = f"token: {token}"
    redacted = redact_secrets(text)
    assert token not in redacted
    assert "[REDACTED]" in redacted
