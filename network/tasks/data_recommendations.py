import numpy
from celery import group
from celery.decorators import task
from django.core.exceptions import ValidationError
from django.db.models import Q
from keras.models import load_model

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
                pca.neural_network_file is not None,
                pca.neural_network_scaler is not None,
            ]):

                nn_model = load_model(pca.neural_network_file.path)

                other_datasets = models.Dataset.objects.filter(
                    assembly=assembly,
                    experiment__experiment_type=exp_type,
                )

                ds_1 = dataset
                for ds_2 in other_datasets:

                    if ds_1 != ds_2 and ds_1.experiment != ds_2.experiment:

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

                            combined_vec_1 = \
                                pca.neural_network_scaler.transform(
                                    [vec_1 + vec_2])
                            combined_vec_2 = \
                                pca.neural_network_scaler.transform(
                                    [vec_2 + vec_1])

                            sim_1 = nn_model.predict_classes(
                                numpy.array(combined_vec_1))[0][0]
                            sim_2 = nn_model.predict_classes(
                                numpy.array(combined_vec_2))[0][0]

                            if bool(sim_1) and bool(sim_2):
                                try:
                                    (models.Similarity
                                           .objects.update_or_create(
                                               experiment_1=ds_1.experiment,
                                               experiment_2=ds_2.experiment,
                                               dataset_1=ds_1,
                                               dataset_2=ds_2,
                                               sim_type='primary',
                                           ))
                                except ValidationError:
                                    pass

                                try:
                                    (models.Similarity
                                           .objects.update_or_create(
                                               experiment_1=ds_2.experiment,
                                               experiment_2=ds_1.experiment,
                                               dataset_1=ds_2,
                                               dataset_2=ds_1,
                                               sim_type='primary',
                                           ))
                                except ValidationError:
                                    pass
                            else:
                                (models.Similarity
                                       .objects.filter(
                                           experiment_1=ds_1.experiment,
                                           experiment_2=ds_2.experiment,
                                           dataset_1=ds_1,
                                           dataset_2=ds_2,
                                           sim_type='primary',
                                       ).delete())
                                (models.Similarity
                                       .objects.filter(
                                           experiment_1=ds_2.experiment,
                                           experiment_2=ds_1.experiment,
                                           dataset_1=ds_2,
                                           dataset_2=ds_1,
                                           sim_type='primary',
                                       ).delete())
