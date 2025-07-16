import os
import menace.chaos_tester as ct


def test_corrupt_disk(tmp_path):
    f = tmp_path / "file"
    f.write_text("hello")
    ct.ChaosTester.corrupt_disk(str(f))
    assert f.exists()


def test_partition_network():
    res = ct.ChaosTester.partition_network(["a", "b", "c"])
    assert isinstance(res, list)
