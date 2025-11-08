from menace_sandbox.celery_app import app


@app.task
def ping():
    return "pong"
