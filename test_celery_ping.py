from celery_app import app


@app.task
def ping():
    return "pong"


if __name__ == "__main__":
    result = ping.delay()
    print("Task sent. Waiting for result...")
    print(result.get(timeout=10))
