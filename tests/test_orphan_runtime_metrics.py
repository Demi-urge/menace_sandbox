import orphan_analyzer

def test_subprocess_and_env_detection(tmp_path):
    mod = tmp_path / "mod.py"
    mod.write_text(
        "import os, subprocess\n"
        "os.environ['X'] = '1'\n"
        "try:\n"
        "    subprocess.Popen(['echo', 'hi'])\n"
        "except Exception:\n"
        "    pass\n"
    )
    metrics = orphan_analyzer._runtime_metrics(mod)
    assert metrics['spawn_attempts'] == 1
    assert metrics['env_writes'] == 1
    assert metrics['exec_success'] is True
