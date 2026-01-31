from menace.error_ontology import ErrorCategory, classify_error


def test_classify_error_by_exception_type():
    err = KeyError("missing")
    result = classify_error(err)
    assert result["data"]["category"] == ErrorCategory.EdgeCaseFailure.value


def test_classify_error_by_phrase():
    assert (
        classify_error("missing config file")["data"]["category"]
        == ErrorCategory.ConfigError.value
    )
    assert (
        classify_error("unhandled exception in worker")["data"]["category"]
        == ErrorCategory.UnhandledException.value
    )


def test_classify_error_bundle_selects_first_non_other():
    inputs = [
        "Traceback (most recent call last):",
        "TypeError: unsupported operand",
    ]
    result = classify_error(inputs)
    assert result["data"]["category"] == ErrorCategory.TypeErrorMismatch.value
