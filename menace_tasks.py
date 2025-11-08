from .celery_app import app


@app.task
def ping():
    return "pong"
