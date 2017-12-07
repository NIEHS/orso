from django.core.management.base import BaseCommand

from network.tasks import set_selected_transcripts_for_genes


class Command(BaseCommand):
    help = '''
        Set the selected transcript for each gene.
    '''

    def handle(self, *args, **options):
        set_selected_transcripts_for_genes()
