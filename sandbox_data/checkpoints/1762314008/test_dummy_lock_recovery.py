import os, time
from lock_utils import _ContextFileLock, is_lock_stale, Timeout


def test_dummy_lock_recovery(tmp_path):
    lock_path = tmp_path / "l"
    lock = _ContextFileLock(str(lock_path))
    # create stale lock
    lock_path.write_text(f"{os.getpid()+1000},{time.time()-2}")
    queue = ["x"]
    recovered = []

    def startup():
        if lock_path.exists() and is_lock_stale(str(lock_path)):
            os.remove(lock_path)
        try:
            lock.acquire(timeout=0)
        except Timeout:
            if lock.is_lock_stale():
                os.remove(lock_path)
                lock.acquire(timeout=0)
            else:
                raise
        try:
            while queue:
                recovered.append(queue.pop(0))
        finally:
            lock.release()

    startup()
    assert recovered == ["x"]
    assert not lock_path.exists()
