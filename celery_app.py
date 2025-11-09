from celery import Celery

app = Celery("menace_sandbox")

app.conf.update(
    broker_url="redis://localhost:6379/0",
    result_backend="redis://localhost:6379/0",
    task_track_started=True,
    timezone="Australia/Brisbane",
    enable_utc=True,
)

app.autodiscover_tasks(["menace_sandbox"])
