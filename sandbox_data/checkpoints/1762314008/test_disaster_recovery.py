import subprocess
import types

from menace.disaster_recovery import DisasterRecovery


def test_backup_and_restore(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    file = data / "x.txt"
    file.write_text("1")
    dr = DisasterRecovery([str(data)], backup_dir=str(tmp_path / "bk"))
    archive = dr.backup()
    assert archive.exists()
    file.unlink()
    monkeypatch.chdir(tmp_path)
    dr.restore(archive)
    assert file.exists()


def test_backup_sync_remote(tmp_path, monkeypatch):
    data = tmp_path / "d"
    data.mkdir()
    (data / "x.txt").write_text("1")
    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("BACKUP_HOSTS", "host")
    dr = DisasterRecovery([str(data)], backup_dir=str(tmp_path / "bk"))
    archive = dr.backup()
    assert ["rsync", str(archive), "host"] in calls


def test_backup_sync_s3(tmp_path, monkeypatch):
    data = tmp_path / "d"
    data.mkdir()
    (data / "y.txt").write_text("1")
    uploaded = {}

    class FakeClient:
        def upload_file(self, src, bucket, key):
            uploaded["bucket"] = bucket
            uploaded["key"] = key

    fake_boto3 = types.SimpleNamespace(client=lambda *_a, **_k: FakeClient())
    import menace.disaster_recovery as dr_mod
    monkeypatch.setattr(dr_mod, "boto3", fake_boto3)
    monkeypatch.setenv("BACKUP_HOSTS", "s3://bkt/path")
    dr = dr_mod.DisasterRecovery([str(data)], backup_dir=str(tmp_path / "bk"))
    archive = dr.backup()
    assert uploaded["bucket"] == "bkt"
    assert uploaded["key"].endswith(archive.name)


def test_missing_boto3_skips_s3_hosts(tmp_path, monkeypatch, caplog):
    data = tmp_path / "d"
    data.mkdir()
    (data / "z.txt").write_text("1")

    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    import menace.disaster_recovery as dr_mod
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(dr_mod, "boto3", None)
    monkeypatch.setenv("BACKUP_HOSTS", "s3://bkt/path,host")

    caplog.set_level("INFO")
    dr = dr_mod.DisasterRecovery([str(data)], backup_dir=str(tmp_path / "bk"))
    assert "boto3 not available" in caplog.text
    caplog.clear()

    archive = dr.backup()
    assert ["rsync", str(archive), "host"] in calls
    assert "Skipping s3://bkt/path" in caplog.text
