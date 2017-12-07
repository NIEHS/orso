"""
WSGI config for genomics_network project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.10/howto/deployment/wsgi/
"""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "genomics_network.settings")
os.environ['LC_ALL'] = 'en_US.UTF-8'

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
