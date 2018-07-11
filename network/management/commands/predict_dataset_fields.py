from django.core.management.base import BaseCommand

from network.tasks.neural_network import predict_all_dataset_fields


class Command(BaseCommand):
    help = '''
        Update predicted dataset fields.
    '''

    def handle(self, *args, **options):
        print('Updating predicted dataset fields...')
        predict_all_dataset_fields()
        print('Done.')
