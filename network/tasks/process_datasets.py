import json
import os

from celery import group
from celery.decorators import task

from analysis import metaplot, transcript_coverage
from analysis.normalization import normalize_locus_intersection_values
from analysis.utils import download_dataset_bigwigs
from network import models


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

        # Remove bigwigs
        for bigwig_paths in downloaded_bigwigs.values():
            for path in bigwig_paths.values():
                if path:
                    os.remove(path)


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
