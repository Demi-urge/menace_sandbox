import ast
import logging
from pathlib import Path

import pytest


def _build_check_permission():
    src = Path("self_coding_engine.py").read_text()
    tree = ast.parse(src)
    class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SelfCodingEngine")
    method = next(m for m in class_node.body if isinstance(m, ast.FunctionDef) and m.name == "_check_permission")
    new_class = ast.ClassDef("SelfCodingEngine", [], [], [method], [])
    module = ast.Module([new_class], type_ignores=[])
    module = ast.fix_missing_locations(module)

    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("menace", Path("__init__.py"))
    menace = importlib.util.module_from_spec(spec)
    menace.__path__ = [str(Path().resolve())]
    sys.modules.setdefault("menace", menace)
    spec.loader.exec_module(menace)
    from menace.access_control import READ, WRITE, check_permission

    ns = {"READ": READ, "WRITE": WRITE, "check_permission": check_permission}
    exec(compile(module, "<ast>", "exec"), ns)
    return ns["SelfCodingEngine"]


def _integrate_insights(engine, description):
    from menace.log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT

    def recent_feedback(svc):
        return "fb"

    def recent_improvement_path(svc):
        return "path"

    def recent_error_fix(svc):
        return "fix"

    insight_lines = []
    if engine.knowledge_service:
        for label, func in [
            (FEEDBACK, recent_feedback),
            (IMPROVEMENT_PATH, recent_improvement_path),
            (ERROR_FIX, recent_error_fix),
        ]:
            insight = func(engine.knowledge_service)
            if insight:
                insight_lines.append(f"{label} insight: {insight}")
    combined = "\n".join(insight_lines)
    if combined:
        engine.logger.info(
            "patch history context",
            extra={"description": description, "history": combined, "tags": [INSIGHT]},
        )
    return combined


def test_permission_checks():
    Engine = _build_check_permission()
    eng = Engine()
    from menace.access_control import READ, WRITE

    eng.bot_roles = {"bob": READ, "alice": WRITE}
    with pytest.raises(PermissionError):
        eng._check_permission("write", "bob")
    eng._check_permission("write", "alice")


def test_insight_integration(caplog):
    engine = type("E", (), {})()
    engine.knowledge_service = object()
    engine.logger = logging.getLogger("SelfCodingEngine")
    with caplog.at_level(logging.INFO):
        hist = _integrate_insights(engine, "desc")
    assert "fb" in hist and "path" in hist and "fix" in hist
    assert any("patch history context" in r.message for r in caplog.records)

