from django.core.management.base import BaseCommand

from network.tasks.update_pca import update_pca


class Command(BaseCommand):
    help = '''
        Update PCA models.
    '''

    def add_arguments(self, parser):
        parser.add_argument('pca_pk', type=int)

    def handle(self, *args, **options):
        update_pca(options['pca_pk'])
