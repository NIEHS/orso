from django.core.management.base import BaseCommand

from network import models
from network.tasks.analysis.pca import set_pca_plot


class Command(BaseCommand):
    help = '''
        Update PCA plots.
    '''

    def handle(self, *args, **options):
        for pca in models.PCA.objects.all():
            set_pca_plot(pca)
