from django.apps import AppConfig


class NetworkConfig(AppConfig):
    name = 'network'

    def ready(self):

        from . import signals  # noqa
