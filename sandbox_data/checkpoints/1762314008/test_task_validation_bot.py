import importlib
import os
import sys
import types
from dataclasses import dataclass
import pytest

sys.modules.pop("menace", None)
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")


@dataclass
class SynthesisTask:
    description: str
    urgency: int
    complexity: int
    category: str


def _load_tvb(use_marshmallow: bool, monkeypatch):
    sys.modules.pop("menace.task_validation_bot", None)
    stub_isb = types.ModuleType("menace.information_synthesis_bot")
    stub_isb.SynthesisTask = SynthesisTask
    monkeypatch.setitem(sys.modules, "menace.information_synthesis_bot", stub_isb)
    if use_marshmallow:
        stub = types.ModuleType("marshmallow")

        class ValidationError(Exception):
            pass

        class _Field:
            def __init__(self, type_, required: bool = False, *a, **k):
                self.type_ = type_
                self.required = required

        class fields:
            Str = lambda required=False, *a, **k: _Field(str, required=required)
            Int = lambda required=False, *a, **k: _Field(int, required=required)

        class Schema:
            def __init__(self) -> None:
                self.fields = {
                    k: v for k, v in self.__class__.__dict__.items() if isinstance(v, _Field)
                }

            def load(self, data):
                for name, field in self.fields.items():
                    if field.required and name not in data:
                        raise ValidationError(name)
                    if name in data and not isinstance(data[name], field.type_):
                        raise ValidationError(name)
                return data

        stub.Schema = Schema
        stub.fields = fields
        stub.ValidationError = ValidationError
        monkeypatch.setitem(sys.modules, "marshmallow", stub)
    else:
        monkeypatch.setitem(sys.modules, "marshmallow", None)
        sys.modules.pop("marshmallow", None)

    module = importlib.import_module("menace.task_validation_bot")
    return importlib.reload(module)


def test_duplicate_removal(monkeypatch):
    tvb = _load_tvb(False, monkeypatch)
    bot = tvb.TaskValidationBot(["goal"])
    monkeypatch.setattr(bot.app, "send_task", lambda *a, **k: None)
    monkeypatch.setattr(bot.socket, "send_json", lambda *a, **k: None)
    tasks = [
        SynthesisTask(description="goal simple task", urgency=1, complexity=1, category="a"),
        SynthesisTask(description="goal simple task", urgency=1, complexity=1, category="a"),
    ]
    res = bot.validate_tasks(tasks)
    assert len(res) == 1


def test_alignment(monkeypatch):
    sent = []
    tvb = _load_tvb(False, monkeypatch)
    bot = tvb.TaskValidationBot(["goal"])
    monkeypatch.setattr(bot.socket, "send_json", lambda *a, **k: None)
    monkeypatch.setattr(bot.app, "send_task", lambda name, kwargs=None: sent.append((name, kwargs)))
    task = SynthesisTask(description="something else", urgency=1, complexity=1, category="a")
    res = bot.validate_tasks([task])
    assert not res
    assert sent


def test_granularity_split(monkeypatch):
    tvb = _load_tvb(False, monkeypatch)
    bot = tvb.TaskValidationBot(["goal"])
    monkeypatch.setattr(bot.app, "send_task", lambda *a, **k: None)
    monkeypatch.setattr(bot.socket, "send_json", lambda *a, **k: None)
    task = SynthesisTask(
        description="goal step one two three four five six seven and goal step two",
        urgency=1,
        complexity=1,
        category="a",
    )
    res = bot.validate_tasks([task])
    assert len(res) == 2


@pytest.mark.parametrize("use_marshmallow", [False, True])
def test_type_validation(monkeypatch, use_marshmallow):
    tvb = _load_tvb(use_marshmallow, monkeypatch)
    sent = []
    bot = tvb.TaskValidationBot(["goal"])
    monkeypatch.setattr(bot.socket, "send_json", lambda *a, **k: None)
    monkeypatch.setattr(bot.app, "send_task", lambda n, kwargs=None: sent.append((n, kwargs)))
    bad = tvb.SynthesisTask(description="goal", urgency="high", complexity=1, category="a")
    res = bot.validate_tasks([bad])
    assert not res
    assert sent
