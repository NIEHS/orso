from celery.decorators import task
from progress.bar import Bar

from analysis import score
from network import models


@task
def update_dataset_data_scores(datasets, quiet=False):
    '''
    Update or create dataset data distance values.
    '''
    updated = set()

    bar_max = 0
    for ds in datasets:
        bar_max += models.Dataset.objects.filter(
            assembly=ds.assembly,
            experiment__experiment_type=ds.experiment.experiment_type,
        ).count()
    bar = Bar('Processing', max=bar_max)

    for ds_1 in datasets:
        for ds_2 in models.Dataset.objects.filter(
            assembly=ds_1.assembly,
            experiment__experiment_type=ds_1.experiment.experiment_type,
        ):

            _ds_1, _ds_2 = sorted([ds_1, ds_2], key=lambda x: x.pk)
            exp_type_1 = _ds_1.experiment.experiment_type
            exp_type_2 = _ds_2.experiment.experiment_type
            if all([
                (_ds_1, _ds_2) not in updated,
                _ds_1 != _ds_2,
                models.PCATransformedValues.objects.filter(
                    dataset=_ds_1,
                    pca__locus_group__group_type=exp_type_1.relevant_regions,
                ).exists(),
                models.PCATransformedValues.objects.filter(
                    dataset=_ds_2,
                    pca__locus_group__group_type=exp_type_2.relevant_regions,
                ).exists(),
            ]):

                distance = score.score_datasets_by_pca_distance(_ds_1, _ds_2)
                models.DatasetDataDistance.objects.update_or_create(
                    dataset_1=_ds_1,
                    dataset_2=_ds_2,
                    defaults={
                        'distance': distance,
                    },
                )
                updated.add((_ds_1, _ds_2))

            bar.next()

    bar.finish()


@task
def update_experiment_data_scores(experiments):
    '''
    Update or create experiment data distance values.
    '''
    updated = set()

    bar_max = 0
    for exp in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp)
        bar_max += models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp.experiment_type,
        ).count()
    bar = Bar('Processing', max=bar_max)

    for exp_1 in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp_1)
        for exp_2 in models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp_1.experiment_type,
        ):

            _exp_1, _exp_2 = sorted([exp_1, exp_2], key=lambda x: x.pk)
            rr_1 = _exp_1.experiment_type.relevant_regions
            rr_2 = _exp_2.experiment_type.relevant_regions
            if all([
                (_exp_1, _exp_2) not in updated,
                _exp_1 != _exp_2,
                models.PCATransformedValues.objects.filter(
                    dataset__experiment=_exp_1,
                    pca__locus_group__group_type=rr_1,
                ).exists(),
                models.PCATransformedValues.objects.filter(
                    dataset__experiment=_exp_2,
                    pca__locus_group__group_type=rr_2,
                ).exists(),
            ]):

                distance = score.score_experiments_by_pca_distance(
                    _exp_1, _exp_2)
                models.ExperimentDataDistance.objects.update_or_create(
                    experiment_1=_exp_1,
                    experiment_2=_exp_2,
                    defaults={
                        'distance': distance,
                    },
                )
                updated.add((_exp_1, _exp_2))

            bar.next()

    bar.finish()
