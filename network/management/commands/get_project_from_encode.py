from django.core.management.base import BaseCommand
from network.tasks.analysis.encode import Encode


class Command(BaseCommand):
    help = '''
        Call the ENCODE API and save metadata for all datasets to a JSON file.
    '''

    def add_arguments(self, parser):
        parser.add_argument('output_json_file', type=str)

    def handle(self, *args, **options):
        encode = Encode()
        encode.get_experiments()
        encode.make_experiment_json(options['output_json_file'])
