from menace.error_ontology import ErrorCategory, classify_error


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
    assert result["data"]["category"] == ErrorCategory.EdgeCaseFailure.value


def test_classify_error_by_phrase():
    result = classify_error("missing config file")
    _assert_schema(result)
    assert result["data"]["category"] == ErrorCategory.ConfigError.value
    result = classify_error("unhandled exception in worker")
    _assert_schema(result)
    assert result["data"]["category"] == ErrorCategory.UnhandledException.value


def test_classify_error_bundle_selects_first_non_other():
    inputs = [
        "Traceback (most recent call last):",
        "TypeError: unsupported operand",
    ]
    result = classify_error(inputs)
    _assert_schema(result)
    assert result["data"]["category"] == ErrorCategory.TypeErrorMismatch.value


def test_deterministic_classification_for_identical_tracebacks():
    trace = "Traceback (most recent call last):\nTypeError: unsupported operand"
    first = classify_error(trace)
    second = classify_error(trace)
    assert first == second


def test_empty_input_returns_other():
    result = classify_error("")
    _assert_schema(result)
    assert result["data"]["category"] == ErrorCategory.Other.value


def test_multi_error_bundle_determinism_first_non_other_wins():
    inputs = ["Traceback (most recent call last):", "missing config file", "type error"]
    result = classify_error(inputs)
    assert result["data"]["category"] == ErrorCategory.ConfigError.value
    swapped = classify_error(["type error", "missing config file"])
    assert swapped["data"]["category"] == ErrorCategory.TypeErrorMismatch.value


def test_unknown_exception_type_returns_other():
    class UnknownError(Exception):
        pass

    result = classify_error(UnknownError("boom"))
    assert result["data"]["category"] == ErrorCategory.Other.value


def test_unrelated_context_does_not_change_literal_match():
    base = "type error: cannot add"
    augmented = f"{base}\nextra unrelated context"
    assert (
        classify_error(base)["data"]["category"]
        == classify_error(augmented)["data"]["category"]
        == ErrorCategory.TypeErrorMismatch.value
    )
