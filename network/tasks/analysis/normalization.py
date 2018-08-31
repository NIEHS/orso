import json

from celery import group
from celery.decorators import task

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


def normalize_locus_intersection_values(loci, locus_values):
    '''
    Take transcription intersection counts and normalize to pseudoTPM for
    genebody, coding regions, and promoter. Add normalized values to dict.

    Here, normalized values:
    TPM = value / (1K / length) / (1M / total sum)
    '''
    normalized_values = dict()
    locus_cpk = dict()  # cpk = counts per kb

    # Set negative values to zero
    for locus in loci:
        locus_values[locus] = max(locus_values[locus], 0)

    for locus in loci:
        length = 0
        for region in locus.regions:
            length += region[1] - region[0] + 1
        locus_cpk[locus] = locus_values[locus] * (1000.0 / length)

    cpk_sum = sum(locus_cpk.values())

    for locus, value in locus_cpk.items():
        normalized_values[locus] = value / (cpk_sum / 1E6)

    return normalized_values
