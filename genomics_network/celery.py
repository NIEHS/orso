from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

from datetime import datetime
from datetime import timedelta
from celery.decorators import task, periodic_task

# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'genomics_network.settings')

app = Celery('genomics_network')

# Using a string here means the worker don't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()


@task(bind=True)
def debug_task():
    now = datetime.now()
    fn = os.path.expanduser('~/Desktop/debug_task.txt')
    with open(fn, 'w+') as f:
        f.write('I did my thing {}'.format(now.isoformat()))


@periodic_task(run_every=timedelta(seconds=5))
def debug_periodic_task():
    now = datetime.now()
    fn = os.path.expanduser('~/Desktop/debug_periodic_task.txt')
    with open(fn, 'w+') as f:
        f.write('I did my thing {}'.format(now.isoformat()))
