from db_write_queue import remove_processed_lines


def test_backup_written(tmp_path):
    qfile = tmp_path / "queue.log"
    qfile.write_text("a\nb\nc\n")
    remove_processed_lines(qfile, 2)
    assert qfile.read_text() == "c\n"
    bak = tmp_path / "queue.log.bak"
    assert bak.read_text() == "a\nb\n"
