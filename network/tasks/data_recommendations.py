from celery import group
from celery.decorators import task

from network import models


def update_all_primary_data_recommendations():
    experiments = models.Experiment.objects.filter(owners=True)

    tasks = []
    for experiment in experiments:
        tasks.append(update_primary_data_recommendations.si(experiment.pk))

    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def update_primary_data_recommendations(experiment_pk):
    experiment = models.Experiment.objects.get(pk=experiment_pk)
    owners = models.MyUser.objects.filter(experiment=experiment)
    datasets = models.Dataset.objects.filter(experiment=experiment)

    for dataset in datasets:

        assembly = dataset.assembly
        exp_type = experiment.experiment_type

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
                    assembly=assembly,
                    experiment__experiment_type=exp_type,
                )

                ds_1 = dataset
                for ds_2 in other_datasets:

                    if ds_1 != ds_2:

                        try:
                            vec_1 = models.PCATransformedValues.objects.get(
                                dataset=ds_1, pca=pca).transformed_values
                        except models.PCATransformedValues.DoesNotExist:
                            print('No transformed values for {}.'.format(
                                ds_1.name))
                            vec_1 = None

                        try:
                            vec_2 = models.PCATransformedValues.objects.get(
                                dataset=ds_2, pca=pca).transformed_values
                        except models.PCATransformedValues.DoesNotExist:
                            print('No transformed values for {}.'.format(
                                ds_2.name))
                            vec_2 = None

                        if vec_1 and vec_2:
                            vec = pca.neural_network_scaler.transform(
                                [vec_1 + vec_2])
                            sim = pca.neural_network.predict(vec)[0]

                            for owner in owners:

                                if bool(sim):
                                    (models.PrimaryDataRec
                                           .objects.update_or_create(
                                               user=owner,
                                               experiment=ds_2.experiment,
                                               dataset=ds_2,
                                               personal_experiment=ds_1.experiment,  # noqa
                                               personal_dataset=ds_1,
                                           ))
                                else:
                                    try:
                                        (models.PrimaryDataRec
                                               .objects.get(
                                                   user=owner,
                                                   experiment=ds_2.experiment,
                                                   dataset=ds_2,
                                                   personal_experiment=ds_1.experiment,  # noqa
                                                   personal_dataset=ds_1,
                                               ).delete())
                                    except models.PrimaryDataRec.DoesNotExist:
                                        pass
