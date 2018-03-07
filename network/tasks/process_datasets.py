import json

from celery import group, chain, chord
from celery.decorators import task

from analysis import metaplot, transcript_coverage
from analysis.normalization import normalize_locus_intersection_values
from analysis.utils import download_dataset_bigwigs, remove_dataset_bigwigs
from network import models
from network.tasks.data_recommendations import \
    update_dataset_primary_data_scores
from network.tasks.metadata_recommendations import \
    update_dataset_metadata_scores


def process_dataset_batch(datasets, chunk=100):

    for i in range(0, len(datasets), chunk):

        index_1 = i
        index_2 = min(i + chunk, len(datasets))
        dataset_chunk = datasets[index_1:index_2]

        download_dataset_bigwigs(dataset_chunk)

        tasks = []
        for dataset in dataset_chunk:
            tasks.append(process_dataset.s(dataset.pk, download=False))

        job = group(tasks)
        results = job.apply_async()
        results.join()


@task
def process_dataset(dataset_pk, download=True):
    update_dataset_metadata_scores.si(dataset_pk).delay()

    if download:
        chain(
            download_bigwigs.si(dataset_pk),
            chord(
                process_dataset_intersections(dataset_pk),
                update_and_clean.si(dataset_pk),
            ),
        )()
    else:
        chord(
            process_dataset_intersections(dataset_pk),
            update_and_clean.si(dataset_pk),
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
        set_pca_transformed_values(dataset, pca)
    update_dataset_primary_data_scores(dataset.pk)

    dataset.processed = True
    dataset.save()

    remove_bigwigs(dataset_pk)


def set_pca_transformed_values(dataset, pca):

    dij = models.DatasetIntersectionJson.objects.get(
        dataset=dataset, locus_group=pca.locus_group)

    order = models.PCALocusOrder.objects.filter(pca=pca).order_by('order')
    loci = [x.locus for x in order]

    intersection_values = json.loads(dij.intersection_values)

    locus_values = dict()
    for val, pk in zip(
        intersection_values['normalized_values'],
        intersection_values['locus_pks']
    ):
        locus_values[pk] = val

    normalized_values = []
    for locus in loci:
        try:
            normalized_values.append(locus_values[locus.pk])
        except IndexError:
            normalized_values.append(0)

    transformed_values = pca.pca.transform([normalized_values])[0]
    models.PCATransformedValues.objects.update_or_create(
        pca=pca,
        dataset=dij.dataset,
        defaults={
            'transformed_values': transformed_values.tolist(),
        },
    )


@task
def remove_bigwigs(dataset_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    remove_dataset_bigwigs([dataset])


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

    models.DatasetIntersectionJson.objects.update_or_create(
        dataset=dataset,
        locus_group=locus_group,
        defaults={
            'intersection_values': json.dumps(intersection_values),
        }
    )


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
