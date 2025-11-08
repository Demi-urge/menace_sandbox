from celery import Celery

app = Celery("menace_tasks")
app.config_from_object({
    "broker_url": "amqp://localhost",
    "result_backend": "rpc://",
    "timezone": "Australia/Brisbane",
    "enable_utc": True,
    "task_track_started": True,
})
