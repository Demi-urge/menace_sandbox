import ast
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

source = (
    Path(__file__).resolve().parents[1] / "sandbox_runner" / "orphan_discovery.py"
).read_text()
module = ast.parse(source)
functions: Dict[str, str] = {}
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name in {"_log_unresolved", "_eval_simple"}:
        functions[node.name] = ast.get_source_segment(source, node)

globals_dict = {
    "ast": ast,
    "Any": Any,
    "Iterable": Iterable,
    "List": List,
    "Dict": Dict,
    "Mapping": Mapping,
    "Sequence": Sequence,
    "Tuple": Tuple,
    "logger": logging.getLogger(__name__),
    "SAFE_CALLS": {},
}


def _resolve_assignment(*_a, **_k):
    return None


globals_dict["_resolve_assignment"] = _resolve_assignment

exec(functions["_log_unresolved"], globals_dict)
exec(functions["_eval_simple"], globals_dict)

_eval_simple = globals_dict["_eval_simple"]


def test_eval_simple_logs_binop_exception(caplog):
    node = ast.parse("'%s' % ()", mode="eval").body
    with caplog.at_level(logging.DEBUG):
        assert _eval_simple(node, {}, 1) is None
    assert any(
        "Unresolved expression" in record.message and "not enough arguments" in record.message
        for record in caplog.records
    )


def test_eval_simple_logs_call_exception(caplog):
    node = ast.parse("'hello'.index('x')", mode="eval").body
    with caplog.at_level(logging.DEBUG):
        assert _eval_simple(node, {}, 1) is None
    assert any(
        "Unresolved expression" in record.message and "substring not found" in record.message
        for record in caplog.records
    )
