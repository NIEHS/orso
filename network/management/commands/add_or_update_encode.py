from django.core.management.base import BaseCommand

from network.tasks import add_or_update_encode


class Command(BaseCommand):
    help = '''
        Queries ENCODE using its REST API.
    '''

    def handle(self, *args, **options):
        add_or_update_encode()
