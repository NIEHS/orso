import numpy
from celery.decorators import task
from progress.bar import Bar

from analysis import score
from network import models


@task
def update_dataset_data_scores(datasets, quiet=False):
    '''
    Update or create dataset data distance values.
    '''
    bar = Bar('Processing', max=len(datasets))

    for ds_1 in datasets:

        dataset_to_score = dict()

        for ds_2 in models.Dataset.objects.filter(
            assembly=ds_1.assembly,
            experiment__experiment_type=ds_1.experiment.experiment_type,
        ):

            exp_type_1 = ds_1.experiment.experiment_type
            exp_type_2 = ds_2.experiment.experiment_type

            if all([
                ds_1 != ds_2,
                models.PCATransformedValues.objects.filter(
                    dataset=ds_1,
                    pca__locus_group__group_type=exp_type_1.relevant_regions,
                ).exists(),
                models.PCATransformedValues.objects.filter(
                    dataset=ds_2,
                    pca__locus_group__group_type=exp_type_2.relevant_regions,
                ).exists(),
            ]):

                distance = score.score_datasets_by_pca_distance(ds_1, ds_2)
                dataset_to_score[ds_2] = distance

        distances = list(dataset_to_score.values())

        average = numpy.mean(distances)
        sd = numpy.std(distances)

        for ds_2, distance in dataset_to_score.items():
            z_score = (distance - average) / sd

            models.DatasetDataDistance.objects.update_or_create(
                dataset_1=ds_1,
                dataset_2=ds_2,
                defaults={
                    'distance': z_score,
                },
            )

        bar.next()

    bar.finish()


@task
def update_experiment_data_scores(experiments):
    '''
    Update or create experiment data distance values.
    '''
    bar = Bar('Processing', max=len(experiments))

    for exp_1 in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp_1)
        for exp_2 in models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp_1.experiment_type,
        ):

            rr_1 = exp_1.experiment_type.relevant_regions
            rr_2 = exp_2.experiment_type.relevant_regions
            if all([
                exp_1 != exp_2,
                models.PCATransformedValues.objects.filter(
                    dataset__experiment=exp_1,
                    pca__locus_group__group_type=rr_1,
                ).exists(),
                models.PCATransformedValues.objects.filter(
                    dataset__experiment=exp_2,
                    pca__locus_group__group_type=rr_2,
                ).exists(),
            ]):

                distance = score.score_experiments_by_pca_distance(
                    exp_1, exp_2)
                models.ExperimentDataDistance.objects.update_or_create(
                    experiment_1=exp_1,
                    experiment_2=exp_2,
                    defaults={
                        'distance': distance,
                    },
                )

        bar.next()

    bar.finish()
