from django.core.management.base import BaseCommand

from network.tasks.neural_network import fit_neural_networks


class Command(BaseCommand):
    help = '''
        Update recommendation scores.
    '''

    def handle(self, *args, **options):
        print('Updating neural networks...')
        fit_neural_networks()
        print('Done.')
