from multiprocessing import Pool
from subprocess import call

from django.core.management.base import BaseCommand

from network import models


def call_update_pca(pca_pk):
    print('Running PCA: {}'.format(str(pca_pk)))
    call([
        'python', 'manage.py', 'update_pca', str(pca_pk),
    ])


def call_update_pca_plot(pca_pk):
    print('Running PCA: {}'.format(str(pca_pk)))
    call([
        'python', 'manage.py', 'update_pca', '--plot_only', str(pca_pk),
    ])


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
        parser.add_argument(
            '--threads',
            action='store',
            dest='threads',
            type=int,
            help='Number of threads to use',
        )

    def handle(self, *args, **options):

        pcas = []
        for lg in models.LocusGroup.objects.all():
            for exp_type in models.ExperimentType.objects.all():
                if models.Dataset.objects.filter(
                    assembly=lg.assembly,
                    experiment__experiment_type=exp_type,
                ).count() >= 3:  # Verify that there are at least 3 datasets
                    pcas.append(models.PCA.objects.get_or_create(
                        locus_group=lg,
                        experiment_type=exp_type,
                    )[0])

        pool = Pool(options['threads'])

        print('Updating PCAs...')
        if options['plot_only']:
            pool.map(call_update_pca_plot, [pca.pk for pca in pcas])
        else:
            pool.map(call_update_pca, [pca.pk for pca in pcas])
        print('Done.')
