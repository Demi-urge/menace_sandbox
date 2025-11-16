import pytest

pytest.importorskip("sqlalchemy")

import menace.menace as mn


def test_schema_creation(tmp_path):
    db = mn.MenaceDB(url=f"sqlite:///{tmp_path / 'menace.db'}")
    names = set(db.meta.tables.keys())
    expected = {
        "models",
        "information",
        "bots",
        "code",
        "errors",
        "discrepancies",
        "enhancements",
        "workflows",
        "bot_models",
        "bot_workflows",
        "bot_enhancements",
        "bot_errors",
        "code_bots",
        "code_enhancements",
        "code_errors",
        "model_workflows",
        "workflow_models",
        "workflow_bots",
        "information_workflows",
        "information_models",
        "enhancement_models",
        "enhancement_bots",
        "enhancement_workflows",
        "error_bots",
        "error_models",
        "error_codes",
        "error_discrepancies",
        "discrepancy_bots",
        "discrepancy_models",
        "model_errors",
        "model_discrepancies",
        "discrepancy_workflows",
        "discrepancy_enhancements",
    }
    assert expected <= names
    fk = next(iter(db.bot_models.c.model_id.foreign_keys))
    assert fk.column.table.name == "models"
    fk_enh = next(iter(db.bot_enhancements.c.enhancement_id.foreign_keys))
    assert fk_enh.column.table.name == "enhancements"
    fk_cb = next(iter(db.code_bots.c.bot_id.foreign_keys))
    assert fk_cb.column.table.name == "bots"
    fk_ce = next(iter(db.code_enhancements.c.enhancement_id.foreign_keys))
    assert fk_ce.column.table.name == "enhancements"
    fk_cerr = next(iter(db.code_errors.c.error_id.foreign_keys))
    assert fk_cerr.column.table.name == "errors"
    fk_wb = next(iter(db.workflow_bots.c.bot_id.foreign_keys))
    assert fk_wb.column.table.name == "bots"
    fk_be = next(iter(db.bot_errors.c.error_id.foreign_keys))
    assert fk_be.column.table.name == "errors"
    fk_em = next(iter(db.enhancement_models.c.model_id.foreign_keys))
    assert fk_em.column.table.name == "models"
    fk_ec = next(iter(db.error_codes.c.code_id.foreign_keys))
    assert fk_ec.column.table.name == "code"
    fk_de = next(iter(db.discrepancy_enhancements.c.enhancement_id.foreign_keys))
    assert fk_de.column.table.name == "enhancements"
    fk_me = next(iter(db.model_errors.c.error_id.foreign_keys))
    assert fk_me.column.table.name == "errors"
    fk_md = next(iter(db.model_discrepancies.c.discrepancy_id.foreign_keys))
    assert fk_md.column.table.name == "discrepancies"
    assert "linked_model_id" not in db.bots.c
    assert "linked_workflow_id" not in db.bots.c
    assert "enhancement_id" not in db.bots.c
    assert "workflow_id" not in db.models.c
    assert "linked_model_id" not in db.workflows.c
    assert "bot_assignments" not in db.workflows.c
    assert "model_id" not in db.information.c
    assert "linked_model_id" not in db.enhancements.c
    assert "linked_workflow_id" not in db.enhancements.c
    assert "linked_bot_id" not in db.enhancements.c
    assert "linked_bot_id" not in db.errors.c
    assert "linked_model_id" not in db.errors.c
    assert "linked_code_id" not in db.errors.c
    assert "linked_discrepancy_id" not in db.errors.c
    assert "linked_model_id" not in db.discrepancies.c
    assert "linked_bot_id" not in db.discrepancies.c
    assert "linked_workflow_id" not in db.discrepancies.c
    assert "linked_enhancement_id" not in db.discrepancies.c
    assert "discovered_in_stage" not in db.discrepancies.c
    assert "linked_bot_id" not in db.code.c
    assert "patch_history" not in db.code.c
    assert "error_logs" not in db.code.c
    assert "discrepancy_flag" in db.models.c
    assert "error_flag" in db.models.c
    assert "estimated_profit" in db.bots.c
    assert "estimated_profit_per_bot" in db.workflows.c
