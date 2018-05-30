from django.core.management.base import BaseCommand

from network import models
from network.tasks.update_pca import set_pca_plot, update_pca


class Command(BaseCommand):
    help = '''
        Update PCA models.
    '''

    def add_arguments(self, parser):
        parser.add_argument(
            '--plot_only',
            action='store_true',
            help='Only update the PCA plot',
        )
        parser.add_argument('pca_pk', type=int)

    def handle(self, *args, **options):
        if options['plot_only']:
            pca = models.PCA.objects.get(pk=options['pca_pk'])
            set_pca_plot(pca)
        else:
            update_pca(options['pca_pk'])
