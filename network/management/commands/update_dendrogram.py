from subprocess import call

from celery.decorators import task
from django.core.management.base import BaseCommand

from network import models
from network.tasks.analysis.dendrogram import update_dendrogram


@task
def call_update_dendrogram(organism_pk, experiment_type_pk, my_user_pk=None):

    cmd = ['python', 'manage.py', 'update_dendrogram']
    if my_user_pk:
        cmd.extend(['--my_user_pk', str(my_user_pk)])
    cmd.extend([str(organism_pk), str(experiment_type_pk)])
    call(cmd)

    organism = models.Organism.objects.get(pk=organism_pk)
    experiment_type = models.ExperimentType.objects.get(pk=experiment_type_pk)
    if my_user_pk:
        my_user = models.MyUser.objects.get(pk=my_user_pk)
        print('Dendrogram updated: {}, {}, {}.'.format(
            organism.name, experiment_type.name, my_user.user.username))
    else:
        print('Dendrogram updated: {}, {}.'.format(
            organism.name, experiment_type.name))


class Command(BaseCommand):
    help = '''
        Update dendrogram model.
    '''

    def add_arguments(self, parser):
        parser.add_argument('organism_pk', type=int)
        parser.add_argument('experiment_type_pk', type=int)

        parser.add_argument(
            '--my_user_pk',
            action='store',
            dest='my_user_pk',
            type=int,
        )

    def handle(self, *args, **options):
        update_dendrogram(
            options['organism_pk'],
            options['experiment_type_pk'],
            my_user_pk=options['my_user_pk'],
        )
