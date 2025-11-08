from celery import shared_task


@shared_task(name="menace_sandbox.menace_tasks.ping")
def ping():
    return "pong"
