from django.core.management.base import BaseCommand

from network.tasks.dendrogram import update_dendrogram


class Command(BaseCommand):
    help = '''
        Update dendrogram model.
    '''

    def add_arguments(self, parser):
        parser.add_argument('organism_pk', type=int)
        parser.add_argument('experiment_type_pk', type=int)

    def handle(self, *args, **options):
        update_dendrogram(
            options['organism_pk'], options['experiment_type_pk'])
