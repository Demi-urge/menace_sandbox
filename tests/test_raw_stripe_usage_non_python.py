import scripts.check_raw_stripe_usage as check

import pytest


@pytest.mark.parametrize(
    "extension, content",
    [
        (".js", "fetch('https://api.stripe.com/v1/charges')"),
        (".ts", "const key = 'sk_test_123';"),
        (".md", "See https://api.stripe.com/v1/customers"),
        (".yaml", "stripe_key: \"sk_test_123\""),
        (".html", '<a href="https://api.stripe.com/v1/payments">'),
        (".json", '{"key": "sk_live_123"}'),
        (".jsx", 'const k = "pk_live_123";'),
        (".tsx", "fetch('https://api.stripe.com/v1/refunds')"),
    ],
)

def test_detects_stripe_in_non_python_files(tmp_path, monkeypatch, capsys, extension, content):
    path = tmp_path / f"bad{extension}"
    path.write_text(content)

    monkeypatch.setattr(check, "_tracked_files", lambda: [path])

    assert check.main() == 1
    captured = capsys.readouterr()
    assert path.name in captured.out
