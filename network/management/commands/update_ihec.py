from django.core.management.base import BaseCommand

from network.tasks.analysis.ihec import add_ihec


class Command(BaseCommand):
    help = '''
        Updates IHEC experiments using metadata JSONs.
    '''

    def add_arguments(self, parser):
        parser.add_argument('json_path', type=str)

    def handle(self, *args, **options):
        add_ihec(options['json_path'])
