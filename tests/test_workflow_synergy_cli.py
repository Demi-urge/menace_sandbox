import json
from pathlib import Path

import pytest

import workflow_synergy_cli as cli
from workflow_synergy_comparator import OverfittingReport, SynergyScores


def _write_specs(tmp_path: Path) -> tuple[Path, Path]:
    spec = {"steps": [{"module": "a"}]}
    a = tmp_path / "a.workflow.json"
    b = tmp_path / "b.workflow.json"
    a.write_text(json.dumps(spec))
    b.write_text(json.dumps(spec))
    return a, b


def _stub_comparator(monkeypatch):
    report = OverfittingReport(low_entropy=False, repeated_modules={})

    class DummyComparator:
        @classmethod
        def compare(cls, a_spec, b_spec):
            return SynergyScores(
                similarity=1.0,
                shared_module_ratio=1.0,
                entropy_a=1.0,
                entropy_b=1.0,
                expandability=1.0,
                efficiency=0.0,
                modularity=0.0,
                aggregate=1.0,
                overfit_a=report,
                overfit_b=report,
            )

        @staticmethod
        def is_duplicate(
            a, b=None, *, similarity_threshold=0.95, entropy_threshold=0.05
        ):
            return True

    monkeypatch.setattr(cli, "WorkflowSynergyComparator", DummyComparator)


def test_cli_outputs_json(tmp_path, monkeypatch, capsys):
    _stub_comparator(monkeypatch)
    a, b = _write_specs(tmp_path)
    cli.cli([str(a), str(b)])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["duplicate"] is True
    assert data["similarity"] == pytest.approx(1.0)
    assert "efficiency" in data
    assert "modularity" in data


def test_cli_writes_file(tmp_path, monkeypatch):
    _stub_comparator(monkeypatch)
    a, b = _write_specs(tmp_path)
    out_file = tmp_path / "out.json"
    cli.cli([str(a), str(b), "--out", str(out_file)])
    data = json.loads(out_file.read_text())
    assert data["aggregate"] == pytest.approx(1.0)
    assert "efficiency" in data
    assert "modularity" in data
