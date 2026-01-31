from menace.error_ontology import ErrorCategory, classify_error, _TOKEN_RULES


def _assert_schema(result):
    assert set(result.keys()) == {"status", "data", "errors", "meta"}
    assert isinstance(result["status"], str)
    assert isinstance(result["data"], dict)
    assert isinstance(result["errors"], list)
    assert isinstance(result["meta"], dict)


def test_classify_error_by_exception_type():
    err = KeyError("missing")
    result = classify_error(err)
    _assert_schema(result)
    assert result["status"] == ErrorCategory.EdgeCaseFailure.value
    assert result["data"]["matched_rule_id"].startswith("exception:")


def test_classify_error_by_phrase():
    result = classify_error("missing config file")
    _assert_schema(result)
    assert result["status"] == ErrorCategory.ConfigError.value
    assert result["data"]["matched_token"] == "missing config"
    result = classify_error("unhandled exception in worker")
    _assert_schema(result)
    assert result["status"] == ErrorCategory.UnhandledException.value
    assert result["data"]["matched_token"] == "unhandled exception"


def test_classify_error_bundle_returns_other_with_item_matches():
    inputs = [
        "Traceback (most recent call last):",
        "TypeError: unsupported operand",
    ]
    result = classify_error(inputs)
    _assert_schema(result)
    assert result["status"] == ErrorCategory.Other.value
    assert result["data"]["bundle"][1]["status"] == ErrorCategory.TypeErrorMismatch.value


def test_deterministic_classification_for_identical_tracebacks():
    trace = "Traceback (most recent call last):\nTypeError: unsupported operand"
    first = classify_error(trace)
    second = classify_error(trace)
    assert first == second


def test_empty_input_returns_other():
    result = classify_error("")
    _assert_schema(result)
    assert result["status"] == ErrorCategory.Other.value


def test_multi_error_bundle_determinism_per_item_only():
    inputs = ["Traceback (most recent call last):", "missing config file", "type error"]
    result = classify_error(inputs)
    assert result["status"] == ErrorCategory.Other.value
    assert [item["status"] for item in result["data"]["bundle"]] == [
        ErrorCategory.Other.value,
        ErrorCategory.ConfigError.value,
        ErrorCategory.TypeErrorMismatch.value,
    ]
    swapped = classify_error(["type error", "missing config file"])
    assert swapped["status"] == ErrorCategory.Other.value
    assert [item["status"] for item in swapped["data"]["bundle"]] == [
        ErrorCategory.TypeErrorMismatch.value,
        ErrorCategory.ConfigError.value,
    ]


def test_unknown_exception_type_returns_other():
    class UnknownError(Exception):
        pass

    result = classify_error(UnknownError("boom"))
    assert result["status"] == ErrorCategory.Other.value


def test_unrelated_context_does_not_change_literal_match():
    base = "type error: cannot add"
    augmented = f"{base}\nextra unrelated context"
    assert (
        classify_error(base)["status"]
        == classify_error(augmented)["status"]
        == ErrorCategory.TypeErrorMismatch.value
    )


def test_literal_phrase_matches_cover_all_fixed_tokens():
    for rule in _TOKEN_RULES:
        token = rule["match"]
        expected = rule["category"].value
        result = classify_error(f"prefix {token} suffix")
        assert result["status"] == expected
        assert result["data"]["matched_token"] == token
        assert result["data"]["matched_rule_id"] == rule["matched_rule"]
