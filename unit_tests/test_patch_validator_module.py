import patch_validator


def test_validate_patch_returns_structured_payload():
    result = patch_validator.validate_patch(
        "value = 1\n",
        "value = 2\n",
        [],
    )

    assert isinstance(result, dict)
    assert set(result.keys()) >= {"status", "data", "errors", "meta"}


def test_validate_patch_handles_invalid_inputs():
    result = patch_validator.validate_patch(  # type: ignore[arg-type]
        123,
        None,
        "not-a-rule-list",
    )

    assert isinstance(result, dict)
    assert result["errors"]

    text_result = patch_validator.validate_patch_text(123)  # type: ignore[arg-type]
    assert isinstance(text_result, dict)
    assert text_result["valid"] is False
    assert text_result["flags"]
