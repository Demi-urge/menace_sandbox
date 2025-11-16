import ast
import re
from pathlib import Path


def test_ask_with_memory_keys_use_module_action():
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"^[\w_]+\.[\w_]+")
    failures = []
    for path in root.rglob("*.py"):  # path-ignore
        parts = set(path.parts)
        if any(p.startswith('.') for p in parts) or 'build' in parts or 'dist' in parts:
            continue
        if path.name == "test_ask_with_memory_lint.py":  # path-ignore
            continue
        try:
            tree = ast.parse(path.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "ask_with_memory" or (
                    isinstance(func, ast.Attribute) and func.attr == "ask_with_memory"
                ):
                    if len(node.args) >= 2:
                        key_node = node.args[1]
                        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                            if not pattern.match(key_node.value):
                                failures.append(f"{path}:{node.lineno}")
                        elif isinstance(key_node, ast.JoinedStr):
                            const_text = "".join(
                                part.value for part in key_node.values if isinstance(part, ast.Constant)  # noqa: E501
                            )  # noqa: E501
                            if "." not in const_text:
                                failures.append(f"{path}:{node.lineno}")
                        else:
                            failures.append(f"{path}:{node.lineno}")
    assert not failures, "ask_with_memory calls missing module.action key: " + "; ".join(failures)
