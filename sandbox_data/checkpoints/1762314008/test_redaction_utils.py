from dynamic_path_router import resolve_path
from redaction_utils import redact_text

FIXTURE_DIR = resolve_path("tests/fixtures/semantic")


def test_redact_text_on_fixtures():
    secret_api = "api_key=1234567890abcdef1234567890abcdef"
    secret_bearer = "Bearer abcdefghijklmnopqrstuvwxyz1234567890"
    secret_private = """-----BEGIN PRIVATE KEY-----\nABCDEF\n-----END PRIVATE KEY-----"""
    secret_aws = "AKIA1234567890ABCDEF"
    secrets = [secret_api, secret_bearer, secret_private, secret_aws]

    for path in FIXTURE_DIR.glob("*.py"):  # path-ignore
        original = path.read_text()
        combined = original + "\n" + "\n".join(secrets)
        redacted = redact_text(combined)
        assert "Database access helpers." in redacted
        for secret in secrets:
            assert secret not in redacted
        assert redacted.count("[REDACTED]") >= len(secrets)
