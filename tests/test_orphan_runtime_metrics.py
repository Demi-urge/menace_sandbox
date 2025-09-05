import orphan_analyzer

def test_subprocess_and_env_detection(tmp_path):
    mod = tmp_path / "mod.py"  # path-ignore
    mod.write_text(
        "import os, subprocess, threading\n"
        "os.environ['X'] = '1'\n"
        "try:\n"
        "    subprocess.run(['echo', 'hi'])\n"
        "except Exception:\n"
        "    pass\n"
        "def noop():\n"
        "    pass\n"
        "try:\n"
        "    threading.Thread(target=noop).start()\n"
        "except Exception:\n"
        "    pass\n"
    )
    metrics = orphan_analyzer._runtime_metrics(mod)
    assert metrics['process_calls'] == 1
    assert metrics['env_writes'] == 1
    assert metrics['threads_started'] == 1
    assert metrics['exec_success'] is True
