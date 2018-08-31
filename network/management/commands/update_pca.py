from subprocess import call

from celery.decorators import task
from django.core.management.base import BaseCommand

from network import models
from network.tasks.analysis.pca import update_pca


@task
def call_update_pca(locus_group_pk, experiment_type_pk):

    call([
        'python', 'manage.py', 'update_pca', str(locus_group_pk),
        str(experiment_type_pk),
    ])

    locus_group = models.LocusGroup.objects.get(pk=locus_group_pk)
    experiment_type = models.ExperimentType.objects.get(pk=experiment_type_pk)

    print('PCA updated: {}, {}, {}'.format(
        experiment_type.name, locus_group.assembly.name,
        locus_group.group_type))


class Command(BaseCommand):
    help = '''
        Update PCA models.
    '''

    def add_arguments(self, parser):
        parser.add_argument('locus_group_pk', type=int)
        parser.add_argument('experiment_type_pk', type=int)

    def handle(self, *args, **options):
        update_pca(options['locus_group_pk'], options['experiment_type_pk'])
