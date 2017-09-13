from django.core.management.base import BaseCommand
from scipy.stats import spearmanr

from network import models


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '--output_plot',
            action='store',
            dest='output_plot',
            type=str,
            help='Create scatter plot of distance values.',
        )

    def handle(self, *args, **options):

        datasets = models.Dataset.objects.all()
        distances = []

        for i, ds_1 in enumerate(datasets):
            for ds_2 in datasets[i + 1:]:
                try:
                    distances.append((
                        models.DatasetDataDistance.objects.get(
                            dataset_1=ds_1, dataset_2=ds_2).distance,
                        models.DatasetMetadataDistance.objects.get(
                            dataset_1=ds_1, dataset_2=ds_2).distance,
                    ))
                except (
                    models.DatasetDataDistance.DoesNotExist,
                    models.DatasetMetadataDistance.DoesNotExist,
                ):
                    pass

        rho, p = spearmanr(distances)
        print('Rho: {}'.format(str(rho)))
        print('P-value: {}'.format(str(p)))
