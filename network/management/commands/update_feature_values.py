from django.core.management.base import BaseCommand

from network.tasks.process_datasets import update_all_feature_values


class Command(BaseCommand):

    def handle(self, *args, **options):
        update_all_feature_values()
