from django.core.management.base import BaseCommand

from network.tasks.network import update_dataset_networks, \
    update_experiment_networks, update_organism_networks


class Command(BaseCommand):

    def handle(self, *args, **options):
        update_dataset_networks()
        update_experiment_networks()
        update_organism_networks()
