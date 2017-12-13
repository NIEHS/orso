import numpy
from celery import group
from celery.decorators import task

from network import models


@task
def process_experiments(experiments):
    set_experiment_intersection_values(experiments)

    for exp in experiments:
        exp.processed = True
        exp.save()


@task
def set_experiment_intersection_values(experiments):
    '''
    For each experiment, set average intersection values.
    '''
    job = group(_set_experiment_intersection_values.s(exp.pk)
                for exp in experiments)
    job.apply_async()


@task
def _set_experiment_intersection_values(experiment_pk):
    '''
    For a given experiment, set the average intersection values.
    '''
    exp = models.Experiment.objects.get(pk=experiment_pk)

    # Remove existing
    models.ExperimentIntersection.objects.filter(experiment=exp).delete()

    # Create new
    exp_intersections = []
    for assembly in \
            models.Assembly.objects.filter(dataset__experiment=exp).distinct():
        loci = models.Locus.objects.filter(group__assembly=assembly)
        for locus in loci:
            intersection_values = models.DatasetIntersection.objects.filter(
                locus=locus, dataset__experiment=exp,
            ).values_list('normalized_value', flat=True)
            exp_intersections.append(
                models.ExperimentIntersection(
                    locus=locus,
                    experiment=exp,
                    average_value=numpy.mean(list(intersection_values)),
                )
            )
    models.ExperimentIntersection.objects.bulk_create(exp_intersections)
