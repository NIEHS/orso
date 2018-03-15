from celery import chain, chord
from celery.decorators import task

from network import models
from network.tasks.metadata_recommendations import \
    update_metadata_sims_and_recs
from network.tasks.process_datasets import (
    download_bigwigs,
    process_dataset_intersections,
    update_and_clean,
)


@task
def process_experiment(experiment_pk):
    update_metadata_sims_and_recs.si(experiment_pk).delay()
    process_experiment_datasets.si(experiment_pk).delay()


@task
def process_experiment_datasets(experiment_pk, download=True):
    datasets = models.Dataset.objects.filter(experiment__pk=experiment_pk)
    dataset_pks = [ds.pk for ds in datasets]
    if download:
        chain(
            download_bigwigs.si(dataset_pks),
            chord(
                process_dataset_intersections(dataset_pks),
                update_and_clean.si(dataset_pks, experiment_pk=experiment_pk),
            ),
        )()
    else:
        chord(
            process_dataset_intersections(dataset_pks),
            update_and_clean.si(dataset_pks, experiment_pk=experiment_pk),
        )()
