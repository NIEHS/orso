from django.core.management.base import BaseCommand

from network.tasks.network import update_dataset_networks, \
    update_experiment_networks, update_organism_networks


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument(
            '--organism',
            action='store_true',
            help='Update organism-level network plots'
        )
        parser.add_argument(
            '--dataset',
            action='store_true',
            help='Update dataset-level network plots'
        )
        parser.add_argument(
            '--experiment',
            action='store_true',
            help='Update experiment-level network plots'
        )

    def handle(self, *args, **options):
        if all([
            not options['organism'],
            not options['dataset'],
            not options['experiment'],
        ]):
            print('No scores indicated for update. Try "--help" for a list of '
                  'appropriate flags.')
        else:
            if options['organism']:
                print('Updating organism-level network plots...')
                update_organism_networks()
                print('Done.')
            if options['dataset']:
                print('Updating dataset-level network plots...')
                update_dataset_networks()
                print('Done.')
            if options['experiment']:
                print('Updating experiment-level network plots...')
                update_experiment_networks()
                print('Done.')
