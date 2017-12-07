from django.core.management.base import BaseCommand

from network import models, tasks


class Command(BaseCommand):
    help = '''
        Update recommendation scores.
    '''

    def add_arguments(self, parser):
        parser.add_argument(
            '--primary',
            action='store_true',
            help='Update primary data-based recommendation scores'
        )
        parser.add_argument(
            '--metadata',
            action='store_true',
            help='Update metadata-based recommendation scores'
        )
        parser.add_argument(
            '--user',
            action='store_true',
            help='Update user interaction-based recommendation scores'
        )

    def handle(self, *args, **options):
        if all([
            not options['primary'],
            not options['metadata'],
            not options['user'],
        ]):
            print('No scores indicated for update. Try "--help" for a list of '
                  'appropriate flags.')
        else:
            datasets = models.Dataset.objects.all()
            experiments = models.Experiment.objects.all()

            if options['primary']:
                print('Processing dataset primary count data...')
                tasks.update_dataset_data_scores(datasets)
                print('Processing experiment primary count data...')
                tasks.update_experiment_data_scores(experiments)
            if options['metadata']:
                print('Processing dataset metadata...')
                tasks.update_dataset_metadata_scores(datasets)
                print('Processing experiment metadata...')
                tasks.update_experiment_metadata_scores(experiments)
            if options['user']:
                pass
