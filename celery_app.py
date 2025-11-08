from celery import Celery

app = Celery(
    "menace",
    broker="amqp://guest:guest@localhost:5672//",
    backend="rpc://",
)

# Config
app.conf.update(
    timezone="Australia/Brisbane",
    enable_utc=True,
    task_track_started=True,
)

# ⬇️ This line is crucial
app.autodiscover_tasks(["menace_sandbox"])

# OR just directly import
import menace_sandbox.menace_tasks  # noqa: E402,F401  # ensure task registration
