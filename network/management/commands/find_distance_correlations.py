import matplotlib
matplotlib.use('TkAgg')  # noqa
import matplotlib.pyplot as plt
from django.core.management.base import BaseCommand
from scipy.stats import spearmanr

from network import models


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('assembly', type=str)
        parser.add_argument('experiment_type', type=str)

        parser.add_argument(
            '--output_plot',
            action='store',
            dest='output_plot',
            type=str,
            help='Create scatter plot of distance values.',
        )

    def handle(self, *args, **options):

        datasets = models.Dataset.objects.filter(
            assembly__name=options['assembly'],
            experiment__experiment_type__name=options['experiment_type'],
        )
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

        if options['output_plot']:
            plt.scatter(
                [x[0] for x in distances], [x[1] for x in distances])
            plt.title('Distance correlation: {}, {}'.format(
                options['assembly'], options['experiment_type']))
            plt.xlabel('Data distances')
            plt.ylabel('Metadata distances')
            plt.savefig(options['output_plot'])

        rho, p = spearmanr(distances)
        print('Rho: {}'.format(str(rho)))
        print('P-value: {}'.format(str(p)))
