from django.core.management.base import BaseCommand

from network.tasks.update_pca import add_or_update_pcas


class Command(BaseCommand):
    help = '''
        Update PCA models.
    '''

    def add_arguments(self, parser):
        parser.add_argument(
            '--threads',
            action='store',
            dest='threads',
            type=int,
            help='Number of threads to use',
        )

    def handle(self, *args, **options):
        add_or_update_pcas(threads=options['threads'])
