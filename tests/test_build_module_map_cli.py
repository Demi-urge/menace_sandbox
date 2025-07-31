import scripts.build_module_map as bm
import scripts.generate_module_map as gm


def test_build_module_map_cli_options(monkeypatch, tmp_path):
    called = {}

    def fake_generate(output, *, root, algorithm, threshold, semantic):
        called.update(
            {
                "output": output,
                "root": root,
                "algorithm": algorithm,
                "threshold": threshold,
                "semantic": semantic,
            }
        )
        return {}

    monkeypatch.setattr(gm, "generate_module_map", fake_generate)

    out = tmp_path / "map.json"
    bm.main([
        str(tmp_path),
        "--output",
        str(out),
        "--algorithm",
        "label",
        "--threshold",
        "0.3",
        "--semantic",
    ])

    assert called["output"] == out
    assert called["root"] == tmp_path
    assert called["algorithm"] == "label"
    assert called["threshold"] == 0.3
    assert called["semantic"] is True
