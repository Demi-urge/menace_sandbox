import json
import sys
from tools.override_cli import main


def test_generate_and_apply(tmp_path, monkeypatch, capsys):
    key = tmp_path / "key"
    key.write_text("secret")
    out = tmp_path / "override.json"
    data = '{"verdict": "promote"}'

    monkeypatch.setattr(sys, "argv", ["prog", "generate", data, str(key), str(out)])
    main()
    capsys.readouterr()
    assert out.exists()

    monkeypatch.setattr(sys, "argv", ["prog", "apply", str(out), str(key)])
    main()
    out_data = json.loads(capsys.readouterr().out)
    assert out_data["valid"] is True
    assert out_data["data"]["verdict"] == "promote"
