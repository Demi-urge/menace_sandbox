from celery import Celery

app = Celery("menace_tasks", broker="amqp://guest:guest@localhost:5672//")

# Optional: backend if you want task results
# app.conf.result_backend = 'rpc://'

# Optional: timezone, task modules
app.conf.update(
    timezone='Australia/Brisbane',
    enable_utc=True,
    task_track_started=True,
)
