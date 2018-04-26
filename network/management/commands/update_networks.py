from django.core.management.base import BaseCommand

from network.tasks.network import update_network_plots


class Command(BaseCommand):

    def handle(self, *args, **options):
        update_network_plots()
