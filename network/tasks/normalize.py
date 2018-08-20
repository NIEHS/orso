import json

from celery import group
from celery.decorators import task

from network.tasks.analysis.normalization import normalize_locus_intersection_values
from network import models


@task()
def normalize_dataset_intersections():
    dataset_intersections = models.DatasetIntersectionJson.objects.filter(
        dataset__processed=True).distinct()

    job = group(_normalize_dataset_intersections.s(intersection.pk)
                for intersection in dataset_intersections)
    job.apply_async()


@task()
def _normalize_dataset_intersections(intersection_pk):
    dij = models.DatasetIntersectionJson.objects.get(pk=intersection_pk)
    intersection_values = json.loads(dij.intersection_values)

    locus_pks = intersection_values['locus_pks']
    loci = list(models.Locus.objects.filter(pk__in=locus_pks))
    loci.sort(key=lambda locus: locus_pks.index(locus.pk))

    locus_values = dict()
    raw_values = intersection_values['raw_values']
    for value, locus in zip(raw_values, loci):
        locus_values[locus] = value

    normalized_values = \
        normalize_locus_intersection_values(loci, locus_values)

    _normalized_values = []
    for locus in loci:
        _normalized_values.append(normalized_values[locus])

    intersection_values['normalized_values'] = _normalized_values

    dij.intersection_values = json.dumps(intersection_values)
    dij.save()
