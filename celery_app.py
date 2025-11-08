import multiprocessing as mp
import os
from typing import Any

from celery import Celery


def _pool_warmup(_: int) -> Any:
    """No-op used to warm up the multiprocessing pool when executed as a script."""
    return None


try:
    mp.set_start_method("spawn")
except RuntimeError:
    # If the start method has already been set elsewhere we only raise when it's
    # incompatible with the desired "spawn" strategy. This mirrors the Windows
    # guidance provided during development.
    if mp.get_start_method(allow_none=True) not in (None, "spawn"):
        raise


app = Celery(
    "menace_sandbox",
    broker=os.getenv("CELERY_BROKER_URL", "memory://"),
)
app.conf.result_backend = os.getenv("CELERY_RESULT_BACKEND", "cache+memory://")

# Config
app.conf.update(
    timezone="Australia/Brisbane",
    enable_utc=True,
    task_track_started=True,
)

# ⬇️ This line is crucial
app.autodiscover_tasks(["menace_sandbox"])

# OR just directly import
from menace_sandbox.menace_tasks import ping  # noqa: E402,F401  # ensure task registration


if __name__ == "__main__":
    cpu_count = os.cpu_count() or 1
    process_count = max(1, cpu_count // 2)
    with mp.Pool(processes=process_count) as pool:
        pool.map(_pool_warmup, range(process_count))
