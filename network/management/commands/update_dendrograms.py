from multiprocessing import Pool
from subprocess import call

from django.core.management.base import BaseCommand

from network import models


def call_update_dendrogram(organism_pk, experiment_type_pk):
    call([
        'python',
        'manage.py',
        'update_dendrogram',
        str(organism_pk),
        str(experiment_type_pk),
    ])


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument(
            '--threads',
            action='store',
            dest='threads',
            type=int,
            help='Number of threads to use',
        )

    def handle(self, *args, **options):

        pks = []
        for org in models.Organism.objects.all():
            for exp_type in models.ExperimentType.objects.all():
                if models.Experiment.objects.filter(
                    dataset__assembly__organism=org,
                    experiment_type=exp_type,
                ).distinct().count() >= 2:
                    pks.append((org.pk, exp_type.pk))

        pool = Pool(options['threads'])

        print('Updating dendrograms...')
        pool.starmap(call_update_dendrogram, pks)
        print('Done.')
