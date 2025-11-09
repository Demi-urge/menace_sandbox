"""Celery tasks exposed by the Menace sandbox package."""

from menace_sandbox.celery_app import app


@app.task(name="menace_sandbox.menace_tasks.ping")
def ping():
    return "pong"
