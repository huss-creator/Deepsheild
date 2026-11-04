"""Celery application. Broker = Redis in production; development/tests run tasks eagerly
(in-process) when no broker is configured. Job STATE lives in the database (AnalysisJob), not in
Celery's result backend, so the API never depends on the broker to report status."""

import os

from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

app = Celery('deepshield')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
