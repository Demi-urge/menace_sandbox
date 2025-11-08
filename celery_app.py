import os

from celery import Celery

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
