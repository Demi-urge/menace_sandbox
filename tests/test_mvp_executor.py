import mvp_executor


def test_execute_generated_code_times_out() -> None:
    stdout, stderr = mvp_executor.execute_generated_code("while True:\n    pass\n")

    assert stdout == ""
    assert "Execution timed out after 5 seconds." in stderr


def test_execute_generated_code_reports_syntax_errors() -> None:
    stdout, stderr = mvp_executor.execute_generated_code("def broken(:\n    pass\n")

    assert stdout == ""
    assert stderr
    assert "SyntaxError" in stderr


def test_execute_generated_code_blocks_os_imports() -> None:
    stdout, stderr = mvp_executor.execute_generated_code("import os\n")

    assert stdout == ""
    assert stderr == "Blocked import: os. Execution was not attempted."


def test_execute_generated_code_blocks_os_submodule_imports() -> None:
    stdout, stderr = mvp_executor.execute_generated_code("import os.path\n")

    assert stdout == ""
    assert stderr == "Blocked import: os.path. Execution was not attempted."


def test_execute_generated_code_cleans_up_temp_dir(tmp_path, monkeypatch) -> None:
    temp_dir = tmp_path / "mvp_executor_temp"
    temp_dir.mkdir()

    def fake_mkdtemp(prefix: str) -> str:
        assert prefix.startswith("mvp_executor_")
        return str(temp_dir)

    monkeypatch.setattr(mvp_executor.tempfile, "mkdtemp", fake_mkdtemp)

    stdout, stderr = mvp_executor.execute_generated_code("print('ok')\n")

    assert stdout.strip() == "ok"
    assert stderr == ""
    assert not temp_dir.exists()


def test_execute_generated_code_handles_empty_input() -> None:
    stdout, stderr = mvp_executor.execute_generated_code("")

    assert stdout == ""
    assert stderr == ""
