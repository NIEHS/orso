import json
import os

from celery import group
from celery.decorators import task
from django.conf import settings

from network import models


def update_network_plots():

    tasks = []

    for org in models.Organism.objects.all():
        for exp_type in models.ExperimentType.objects.all():
            if models.Experiment.objects.filter(
                dataset__assembly__organism=org,
                experiment_type=exp_type,
            ).exists():
                tasks.append(update_network_plot.si(org.pk, exp_type.pk))

    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def update_network_plot(organism_pk, exp_type_pk):

    organism = models.Organism.objects.get(pk=organism_pk)
    exp_type = models.ExperimentType.objects.get(pk=exp_type_pk)

    experiments = models.Experiment.objects.filter(
        dataset__assembly__organism=organism,
        experiment_type=exp_type)
    similarities = models.Similarity.objects.filter(
        experiment_1__in=experiments,
        experiment_2__in=experiments)

    nodes = []
    edges = []

    with open(os.path.join(settings.COLOR_KEY_DIR, 'cell_type.json')) as f:
        cell_type_to_color = json.load(f)
    with open(os.path.join(settings.COLOR_KEY_DIR, 'target.json')) as f:
        target_to_color = json.load(f)

    color_by_target_group = [
        'ChIP-seq',
        'siRNA knockdown followed by RNA-seq',
        'shRNA knockdown followed by RNA-seq',
        'CRISPR genome editing followed by RNA-seq',
        'CRISPRi followed by RNA-seq',
    ]

    exp_to_nodes = dict()
    for i, exp in enumerate(experiments):
        exp_to_nodes[exp.pk] = i

        try:
            if exp_type.name in color_by_target_group:
                color = target_to_color[exp.target]
            else:
                color = cell_type_to_color[exp.cell_type]
        except KeyError:
            color = '#A9A9A9'

        nodes.append({
            'id': i,
            'title': exp.name,
            'color': color,
        })

    _edges = set()
    for sim in similarities:
        pk_1, pk_2 = sorted((sim.experiment_1.pk, sim.experiment_2.pk))
        _edges.add((pk_1, pk_2))
    for _edge in _edges:
        if _edge[0] != _edge[1]:
            edges.append({
                'from': exp_to_nodes[_edge[0]],
                'to': exp_to_nodes[_edge[1]],
            })

    network = json.dumps({
        'nodes': nodes,
        'edges': edges,
    })

    models.Network.objects.update_or_create(
        organism=organism,
        experiment_type=exp_type,
        defaults={
            'network_plot': network,
        },
    )
