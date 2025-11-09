from celery import Celery
import os

app = Celery('menace_sandbox')
app.conf.broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
app.conf.result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

app.autodiscover_tasks(['menace_sandbox'])
