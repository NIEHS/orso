from django.core.management.base import BaseCommand
from django.db.models import Q

from network import models
from network.tasks import add_or_update_pca


class Command(BaseCommand):
    help = '''
        Update PCA models.
    '''

    def add_arguments(self, parser):
        parser.add_argument(
            '--project_names',
            action='store',
            dest='project_names',
            type=str,
            help='Comma-delimited list of project names to consider',
        )
        parser.add_argument(
            '--project_ids',
            action='store',
            dest='project_ids',
            type=str,
            help='Comma-delimited list of project IDs to consider',
        )

    def handle(self, *args, **options):
        if options['project_names'] or options['project_ids']:
            query = Q()
            if options['project_names']:
                for name in options['project_names'].split(','):
                    query |= Q(experiment__project__name=name)
            if options['project_ids']:
                for pk in options['project_ids'].split(','):
                    query |= Q(experiment__project__pk=pk)
            datasets = models.Dataset.objects.filter(query)
        else:
            datasets = models.Dataset.objects.all()

        add_or_update_pca(datasets)
