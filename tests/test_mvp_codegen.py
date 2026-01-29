import mvp_codegen


class DummyPrompt:
    def __init__(self, user, system, metadata):
        self.user = user
        self.system = system
        self.metadata = metadata


def make_wrapper(output=None, error=None):
    class DummyWrapper:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def generate(self, prompt_obj, context_builder=None, max_new_tokens=256, do_sample=False):
            if error is not None:
                raise error
            return output

    return DummyWrapper


def test_run_generation_without_objective_returns_fallback():
    result = mvp_codegen.run_generation({})
    assert isinstance(result, str)
    assert "Safe fallback script" in result
    assert "No valid code was generated." in result


def test_run_generation_wrapper_failure_returns_fallback(monkeypatch):
    monkeypatch.setattr(mvp_codegen.local_model_wrapper, "LocalModelWrapper", make_wrapper(error=RuntimeError("boom")))
    monkeypatch.setattr(mvp_codegen.local_model_wrapper, "Prompt", DummyPrompt)

    result = mvp_codegen.run_generation({"model": object(), "tokenizer": object(), "objective": "test"})

    assert "Safe fallback script" in result
    assert "No valid code was generated." in result


def test_run_generation_removes_blacklisted_imports(monkeypatch):
    output = "import os\nimport subprocess\nprint('ok')"
    monkeypatch.setattr(mvp_codegen.local_model_wrapper, "LocalModelWrapper", make_wrapper(output=output))
    monkeypatch.setattr(mvp_codegen.local_model_wrapper, "Prompt", DummyPrompt)

    result = mvp_codegen.run_generation({"model": object(), "tokenizer": object(), "objective": "test"})

    assert "import os" not in result
    assert "import subprocess" not in result
    assert "print('ok')" in result


def test_run_generation_truncates_oversized_output(monkeypatch):
    line = "print('x')\n"
    output = line * 600
    monkeypatch.setattr(mvp_codegen.local_model_wrapper, "LocalModelWrapper", make_wrapper(output=output))
    monkeypatch.setattr(mvp_codegen.local_model_wrapper, "Prompt", DummyPrompt)

    result = mvp_codegen.run_generation({"model": object(), "tokenizer": object(), "objective": "test"})

    assert len(result) <= 4000
    assert result.startswith("print('x')")


def test_run_generation_returns_plain_python_code(monkeypatch):
    output = "```python\nprint('hi')\n```"
    monkeypatch.setattr(mvp_codegen.local_model_wrapper, "LocalModelWrapper", make_wrapper(output=output))
    monkeypatch.setattr(mvp_codegen.local_model_wrapper, "Prompt", DummyPrompt)

    result = mvp_codegen.run_generation({"model": object(), "tokenizer": object(), "objective": "test"})

    assert "```" not in result
    assert not result.strip().startswith("{")
    assert not result.strip().endswith("}")
    assert result.strip() == "print('hi')"
