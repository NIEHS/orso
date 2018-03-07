from django.core.management.base import BaseCommand

from network.tasks.data_recommendations import update_primary_data_scores
from network.tasks.metadata_recommendations import update_metadata_scores
from network.tasks.user_recommendations import \
    update_user_based_recommendations


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
            if options['primary']:
                print('Updating primary data-based dataset recommendations...')
                update_primary_data_scores()
            if options['metadata']:
                print('Updating metadata-based dataset recommendations...')
                update_metadata_scores()
            if options['user']:
                print('Updating user-based experiment recommendations...')
                update_user_based_recommendations()
