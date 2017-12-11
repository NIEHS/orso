from django.core.management.base import BaseCommand

from network.tasks.update_pca import add_or_update_pca_transformed_values


class Command(BaseCommand):
    help = '''
        Update PCA tranformed dataset values.
    '''

    def handle(self, *args, **options):
        add_or_update_pca_transformed_values()
