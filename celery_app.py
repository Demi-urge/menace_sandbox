from celery import Celery

app = Celery("menace_sandbox")

app.conf.update(
    broker_url="amqp://guest:guest@localhost:5672//",
    result_backend="redis://localhost:6379/0",
    task_track_started=True,
    timezone="Australia/Brisbane",
    enable_utc=True,
)

app.autodiscover_tasks(["menace_sandbox"])

# Load tasks so Celery can register them
import menace_sandbox.menace_tasks  # noqa: E402  pylint: disable=wrong-import-position
