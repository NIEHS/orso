from celery import group
from celery.decorators import task
from django.db.models import Q

from network import models
from network.tasks.recommendations import update_recommendations


def update_all_primary_data_sims_and_recs():
    all_experiments = models.Experiment.objects.all()

    tasks = []
    for experiment in all_experiments:
        tasks.append(update_primary_data_similarities.si(experiment.pk))

    job = group(tasks)
    results = job.apply_async()
    results.join()

    user_experiments = models.Experiment.objects.filter(
        Q(owners=True) | Q(favorite__user=True))

    tasks = []
    for experiment in user_experiments:
        tasks.append(update_recommendations.si(
            experiment.pk, sim_types=['primary']))

    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def update_primary_data_sims_and_recs(experiment_pk):
    update_primary_data_similarities(experiment_pk)
    update_recommendations(experiment_pk, sim_types=['primary'], lock=False)


@task
def update_primary_data_similarities(experiment_pk):
    experiment = models.Experiment.objects.get(pk=experiment_pk)
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

                            if bool(sim):
                                (models.Similarity
                                       .objects.update_or_create(
                                           experiment_1=ds_1.experiment,
                                           experiment_2=ds_2.experiment,
                                           dataset_1=ds_1,
                                           dataset_2=ds_2,
                                           sim_type='primary',
                                       ))

                                (models.Similarity
                                       .objects.update_or_create(
                                           experiment_1=ds_2.experiment,
                                           experiment_2=ds_1.experiment,
                                           dataset_1=ds_2,
                                           dataset_2=ds_1,
                                           sim_type='primary',
                                       ))
                            else:
                                try:
                                    (models.Similarity
                                           .objects.get(
                                               experiment_1=ds_1.experiment,
                                               experiment_2=ds_2.experiment,
                                               dataset_1=ds_1,
                                               dataset_2=ds_2,
                                               sim_type='primary',
                                           ).delete())
                                except models.Similarity.DoesNotExist:
                                    pass

                                try:
                                    (models.Similarity
                                           .objects.get(
                                               experiment_1=ds_2.experiment,
                                               experiment_2=ds_1.experiment,
                                               dataset_1=ds_2,
                                               dataset_2=ds_1,
                                               sim_type='primary',
                                           ).delete())
                                except models.Similarity.DoesNotExist:
                                    pass
