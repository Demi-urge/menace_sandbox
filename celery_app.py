from celery import Celery

app = Celery(
    "menace",
    broker="amqp://guest:guest@localhost:5672//",
    backend="rpc://",
)

app.autodiscover_tasks(["menace_sandbox"])

import menace_sandbox.menace_tasks  # noqa: E402,F401  # ensure task registration
