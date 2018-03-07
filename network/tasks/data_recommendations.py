from celery import group
from celery.decorators import task

from network import models


def update_primary_data_scores():
    datasets = models.Dataset.objects.all()

    tasks = []
    for dataset in datasets:
        tasks.append(update_dataset_primary_data_scores.s(dataset.pk))

    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def update_dataset_primary_data_scores(dataset_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    assembly = dataset.assembly
    exp_type = dataset.experiment.experiment_type

    try:
        pca = models.PCA.objects.get(
            locus_group__assembly=assembly,
            experiment_type=exp_type,
            locus_group__group_type=exp_type.relevant_regions,
        )
    except models.PCA.DoesNotExist:
        pass
    else:

        if all([
            pca.neural_network is not None,
            pca.neural_network_scaler is not None,
        ]):

            other_datasets = models.Dataset.objects.filter(
                assembly=dataset.assembly,
                experiment__experiment_type=dataset.experiment.experiment_type,
            )

            ds_1 = dataset
            for ds_2 in other_datasets:

                assembly_1 = ds_1.assembly
                exp_type_1 = ds_1.experiment.experiment_type

                assembly_2 = ds_2.assembly
                exp_type_2 = ds_2.experiment.experiment_type

                if all([
                    ds_1 != ds_2,
                    assembly_1 == assembly_2,
                    exp_type_1 == exp_type_2,
                ]):

                    try:
                        vec_1 = models.PCATransformedValues.objects.get(
                            dataset=ds_1, pca=pca).transformed_values
                    except models.PCATransformedValues.DoesNotExist:
                        print('Transformed values do not exist for {}.'.format(
                            ds_1.name))
                        vec_1 = None

                    try:
                        vec_2 = models.PCATransformedValues.objects.get(
                            dataset=ds_2, pca=pca).transformed_values
                    except models.PCATransformedValues.DoesNotExist:
                        print('Transformed values do not exist for {}.'.format(
                            ds_2.name))
                        vec_2 = None

                    if vec_1 and vec_2:
                        vec = pca.neural_network_scaler.transform(
                            [vec_1 + vec_2])
                        sim = pca.neural_network.predict(vec)[0]

                        if ds_1.pk < ds_2.pk:
                            _ds_1, _ds_2 = ds_1, ds_2
                        else:
                            _ds_1, _ds_2 = ds_2, ds_1
                        models.DatasetDataDistance.objects.update_or_create(
                            dataset_1=_ds_1,
                            dataset_2=_ds_2,
                            defaults={
                                'distance': sim,
                            },
                        )
