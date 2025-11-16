import ast
from menace.codex_output_analyzer import flag_unsafe_patterns, Severity


def test_eval_on_input_high_severity():
    tree = ast.parse("eval(input())")
    flags = flag_unsafe_patterns(tree)
    msgs = {f['message']: f['severity'] for f in flags}
    assert "eval on input()" in msgs
    assert msgs["eval on input()"] == Severity.HIGH.value
