import json
import math
import os

import numpy as np
import pandas as pd
from celery import group, chain, chord
from celery.decorators import task
from scipy.spatial.distance import euclidean

from analysis import metaplot, transcript_coverage
from analysis.normalization import normalize_locus_intersection_values
from analysis.utils import (download_dataset_bigwigs, remove_dataset_bigwigs,
                            generate_pca_transformed_df)
from network import models
from network.tasks.update_pca import (
    _add_or_update_pca_transformed_values_json, set_pca_plot)
from network.tasks.metadata_recommendations import \
    update_dataset_metadata_scores


def process_dataset_batch(datasets, chunk=100):

    for i in range(0, len(datasets), chunk):

        index_1 = i
        index_2 = min(i + chunk, len(datasets))
        dataset_chunk = datasets[index_1:index_2]

        downloaded_bigwigs = download_dataset_bigwigs(dataset_chunk)

        tasks = []
        for dataset in dataset_chunk:

            for lg in models.LocusGroup.objects.filter(
                    assembly=dataset.assembly):

                tasks.append(
                    update_or_create_dataset_intersection.s(dataset.pk, lg.pk))
                tasks.append(
                    update_or_create_dataset_metaplot.s(dataset.pk, lg.pk))

        job = group(tasks)
        results = job.apply_async()
        results.join()

        for dataset in dataset_chunk:
            dataset.processed = True
            dataset.save()

        # Remove bigwigs
        for bigwig_paths in downloaded_bigwigs.values():
            for path in bigwig_paths.values():
                if path:
                    os.remove(path)


def process_dataset(dataset):
    chain(
        download_bigwigs.si(dataset.pk),
        chord(
            process_dataset_intersections(dataset.pk),
            update_and_clean.si(dataset.pk),
        ),
    )()


@task
def download_bigwigs(dataset_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    download_dataset_bigwigs([dataset])


def process_dataset_intersections(dataset_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    tasks = []

    for lg in models.LocusGroup.objects.filter(
            assembly=dataset.assembly):

        tasks.append(
            update_or_create_dataset_intersection.si(dataset.pk, lg.pk))
        tasks.append(
            update_or_create_dataset_metaplot.si(dataset.pk, lg.pk))

    return group(tasks)


@task
def update_and_clean(dataset_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    assembly = dataset.assembly
    experiment_type = dataset.experiment.experiment_type

    for pca in models.PCA.objects.filter(
        locus_group__assembly=assembly,
        experiment_type=experiment_type,
    ):
        set_pca_plot(pca)
        if pca.locus_group.group_type == experiment_type.relevant_regions:
            update_data_recommendation_scores(pca.pk)

    update_dataset_metadata_scores([dataset])

    remove_bigwigs(dataset_pk)


@task
def update_data_recommendation_scores(pca_pk):

    pca = models.PCA.objects.get(pk=pca_pk)
    transformed_df = generate_pca_transformed_df(pca)

    correlation_values = []
    for name_1, values_1 in transformed_df.iteritems():
        correlation_values.append([])
        for name_2, values_2 in transformed_df.iteritems():
            if name_1 == name_2:
                correlation_values[-1].append(float('nan'))
            else:
                correlation_values[-1].append(euclidean(values_1, values_2))

    correlation_df = pd.DataFrame(
        data=correlation_values,
        index=transformed_df.columns.values,
        columns=transformed_df.columns.values,
    )

    for pk_1, values_1 in correlation_df.iteritems():
        dataset_1 = models.Dataset.objects.get(pk=pk_1)

        mean = np.nanmean(values_1)
        sd = np.nanstd(values_1)

        for pk_2, value in zip(list(correlation_df.index), values_1):
            if not math.isnan(value):

                dataset_2 = models.Dataset.objects.get(pk=pk_2)

                z_score = (value - mean) / sd
                models.DatasetDataDistance.objects.update_or_create(
                    dataset_1=dataset_1,
                    dataset_2=dataset_2,
                    defaults={
                        'distance': z_score,
                    },
                )


@task
def get_intersection_group(dataset_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    tasks = []

    for lg in models.LocusGroup.objects.filter(
            assembly=dataset.assembly):

        tasks.append(
            update_or_create_dataset_intersection.si(dataset.pk, lg.pk))
        tasks.append(
            update_or_create_dataset_metaplot.si(dataset.pk, lg.pk))

    return group(tasks)


@task
def remove_bigwigs(dataset_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    remove_dataset_bigwigs([dataset])


@task
def set_dataset_as_processed(dataset_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    dataset.processed = True
    dataset.save()


@task
def get_pca_transform_group(dataset_pk):
    print('Running get_pca_transform_group with {}'.format(str(dataset_pk)))
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    tasks = []

    for pca in models.PCA.objects.filter(
        experiment_type=dataset.experiment.experiment_type,
        locus_group__assembly=dataset.assembly,
    ):
        for dij in models.DatasetIntersectionJson.objects.filter(
            dataset=dataset,
            locus_group=pca.locus_group,
        ):
            tasks.append(
                _add_or_update_pca_transformed_values_json.si(dij.pk, pca.pk))

    print('Adding {} tasks'.format(str(len(tasks))))

    return group(tasks)


# @task
# def update_data_recommendation_scores(dataset_pk):
#     print('Finished!')


@task
def update_or_create_dataset_intersection(dataset_pk, locus_group_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    locus_group = models.LocusGroup.objects.get(pk=locus_group_pk)

    lg_bed_path = locus_group.intersection_bed_path
    bigwig_paths = dataset.generate_local_bigwig_paths()

    loci = models.Locus.objects.filter(group=locus_group).order_by('pk')
    locus_values = transcript_coverage.get_locus_values(
        loci,
        lg_bed_path,
        ambiguous_bigwig=bigwig_paths['ambiguous'],
        plus_bigwig=bigwig_paths['plus'],
        minus_bigwig=bigwig_paths['minus'],
    )
    normalized_values = normalize_locus_intersection_values(loci, locus_values)

    intersection_values = {
        'locus_pks': [],
        'raw_values': [],
        'normalized_values': [],
    }
    for locus in loci:
        intersection_values['locus_pks'].append(locus.pk)
        intersection_values['raw_values'].append(locus_values[locus])
        intersection_values['normalized_values'].append(
            normalized_values[locus])

    dij = models.DatasetIntersectionJson.objects.update_or_create(
        dataset=dataset,
        locus_group=locus_group,
        defaults={
            'intersection_values': json.dumps(intersection_values),
        }
    )[0]
    pca = models.PCA.objects.get(
        locus_group=locus_group,
        experiment_type=dataset.experiment.experiment_type
    )
    _add_or_update_pca_transformed_values_json(dij.pk, pca.pk)


@task
def update_or_create_dataset_metaplot(dataset_pk, locus_group_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    locus_group = models.LocusGroup.objects.get(pk=locus_group_pk)

    lg_bed_path = locus_group.metaplot_bed_path
    bigwig_paths = dataset.generate_local_bigwig_paths()

    metaplot_out = metaplot.get_metaplot_values(
        locus_group,
        bed_path=lg_bed_path,
        ambiguous_bigwig=bigwig_paths['ambiguous'],
        plus_bigwig=bigwig_paths['plus'],
        minus_bigwig=bigwig_paths['minus'],
    )

    models.MetaPlot.objects.update_or_create(
        dataset=dataset,
        locus_group=locus_group,
        defaults={
            'metaplot': json.dumps(metaplot_out),
        },
    )
