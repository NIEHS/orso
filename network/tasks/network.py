import json
import math
import os
import random

import networkx as nx
from celery import group
from celery.decorators import task
from django.conf import settings
from django.core.cache import cache
from django.db.models import Q
from fa2 import ForceAtlas2

from network import models

TARGET_RELEVANT_EXP_TYPES = [
    'ChIP-seq',
    'siRNA knockdown followed by RNA-seq',
    'shRNA knockdown followed by RNA-seq',
    'CRISPR genome editing followed by RNA-seq',
    'CRISPRi followed by RNA-seq',
]


def hex_to_rgba(hex_value, alpha=1.0):
    red, green, blue = bytes.fromhex(hex_value[1:])
    return (red, green, blue, alpha)


def blend_rgba(rgba_1, rgba_2):

    def blend(val_1, val_2):
        return math.floor(
            math.pow(
                math.pow(val_1, 2.2) + math.pow(val_2, 2.2),
                1 / 2.2
            )
        )

    red = blend(rgba_1[0], rgba_2[0])
    green = blend(rgba_1[1], rgba_2[1])
    blue = blend(rgba_1[2], rgba_2[2])

    alpha = (rgba_1[3] + rgba_2[3]) / 2

    return (red, green, blue, alpha)


def get_exp_color(experiment):

    def retrieve_color_guide(target_color_guide, json_path):
        guide = cache.get(target_color_guide, None)
        if guide:
            return guide
        else:
            with open(json_path) as f:
                guide = json.load(f)
            cache.set(target_color_guide, guide)
            return guide

    if experiment.experiment_type.name in TARGET_RELEVANT_EXP_TYPES:
        color_guide = retrieve_color_guide(
            'target_color_guide',
            os.path.join(settings.COLOR_KEY_DIR, 'target.json'))
        color_key = 'target'
    else:
        color_guide = retrieve_color_guide(
            'cell_type_color_guide',
            os.path.join(settings.COLOR_KEY_DIR, 'cell_type.json'))
        color_key = 'cell_type'

    try:
        color = color_guide[getattr(experiment, color_key)]
    except KeyError:
        color = '#A9A9A9'

    return color


class Network:

    def __init__(self, node_objects_list, edge_objects_list):
        self.node_objects_list = node_objects_list
        self.edge_objects_list = edge_objects_list

    def get_node_pk(self, obj):
        return obj.pk

    def get_node_name(self, obj):
        return obj.name

    def get_node_color(self, obj):
        return get_exp_color(obj)

    def check_node_equivalence(self, obj_1, obj_2):
        return obj_1 == obj_2

    def get_edge_node_1_pk(self, obj):
        return obj.experiment_1.pk

    def get_edge_node_2_pk(self, obj):
        return obj.experiment_2.pk

    def create_network_json(self):

        edges = set()
        for obj in self.edge_objects_list:
            edges.add(tuple(sorted([
                self.get_edge_node_1_pk(obj),
                self.get_edge_node_2_pk(obj),
            ])))

        g = nx.Graph()

        for i, obj in enumerate(self.node_objects_list):
            g.add_node(i)

        for i, obj_1 in enumerate(self.node_objects_list):
            for j, obj_2 in enumerate(self.node_objects_list):
                if any([
                    tuple(sorted([
                        self.get_node_pk(obj_1),
                        self.get_node_pk(obj_2),
                    ])) in edges,
                    self.check_node_equivalence(obj_1, obj_2)
                ]):
                    g.add_edge(i, j)

        fa2 = ForceAtlas2()
        try:
            positions = fa2.forceatlas2_networkx_layout(
                g, pos=None, iterations=2000)
        except ZeroDivisionError:
            positions = dict()
            for i in range(len(self.node_objects_list)):
                positions[i] = (random.random(), random.random())

        node_list = []

        for i, obj in enumerate(self.node_objects_list):
            position = positions[i]

            hex_color = self.get_node_color(obj)
            rgba = hex_to_rgba(hex_color)

            node_list.append({
                'id': i,
                'label': self.get_node_name(obj),
                'color': 'rgba({}, {}, {}, {})'.format(
                    *[str(n) for n in rgba]),
                'x': position[0],
                'y': position[1],
                'size': 8,
            })

        edge_list = []
        edge_count = 0

        for i, obj_1 in enumerate(self.node_objects_list):
            for j, obj_2 in enumerate(self.node_objects_list):
                if any([
                    tuple(sorted([
                        self.get_node_pk(obj_1),
                        self.get_node_pk(obj_2),
                    ])) in edges,
                    self.check_node_equivalence(obj_1, obj_2)
                ]):

                    hex_1 = self.get_node_color(obj_1)
                    rgba_1 = hex_to_rgba(hex_1)

                    hex_2 = self.get_node_color(obj_2)
                    rgba_2 = hex_to_rgba(hex_2)

                    rgba = blend_rgba(rgba_1, rgba_2)

                    edge_list.append({
                        'id': edge_count,
                        'source': i,
                        'target': j,
                        'color': 'rgba({}, {}, {}, {})'.format(
                            *[str(n) for n in rgba]),
                    })
                    edge_count += 1

        network = json.dumps({
            'nodes': node_list,
            'edges': edge_list,
        })

        return network


def update_experiment_networks():

    tasks = []

    for exp in models.Experiment.objects.all():
        tasks.append(update_experiment_network.si(exp.pk))

    job = group(tasks)
    results = job.apply_async()
    results.join()


def update_dataset_networks():

    tasks = []

    for ds in models.Dataset.objects.all():
        tasks.append(update_dataset_network.si(ds.pk))

    job = group(tasks)
    results = job.apply_async()
    results.join()


def update_organism_networks():

    tasks = []

    for org in models.Organism.objects.all():
        for exp_type in models.ExperimentType.objects.all():
            if models.Experiment.objects.filter(
                dataset__assembly__organism=org,
                experiment_type=exp_type,
            ).exists():
                tasks.append(update_organism_network.si(org.pk, exp_type.pk))

    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def update_experiment_network(experiment_pk):

    experiment = models.Experiment.objects.get(pk=experiment_pk)

    connected_experiments = models.Experiment.objects.filter(
        Q(sim_experiment_2__experiment_1=experiment) |
        Q(pk=experiment_pk)).distinct()
    similarities = models.Similarity.objects.filter(
        experiment_1__in=connected_experiments)

    network = Network(connected_experiments, similarities)
    network_json = network.create_network_json()

    models.ExperimentNetwork.objects.update_or_create(
        experiment=experiment,
        defaults={
            'network_plot': network_json,
        },
    )


@task
def update_dataset_network(dataset_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    experiment = dataset.experiment

    connected_experiments = models.Experiment.objects.filter(
        Q(sim_experiment_2__experiment_1=experiment) |
        Q(pk=experiment.pk)).distinct()
    connected_datasets = models.Dataset.objects.filter(
        experiment__in=connected_experiments).distinct()
    similarities = models.Similarity.objects.filter(
        experiment_1__in=connected_experiments)

    def get_node_pk(obj):
        return obj.experiment.pk

    def get_node_color(obj):
        return get_exp_color(obj.experiment)

    def check_node_equivalence(obj_1, obj_2):
        return obj_1.experiment == obj_2.experiment

    network = Network(connected_datasets, similarities)
    network.get_node_pk = get_node_pk
    network.get_node_color = get_node_color
    network.check_node_equivalence = check_node_equivalence
    network_json = network.create_network_json()

    models.DatasetNetwork.objects.update_or_create(
        dataset=dataset,
        defaults={
            'network_plot': network_json,
        },
    )


@task
def update_organism_network(organism_pk, exp_type_pk):

    organism = models.Organism.objects.get(pk=organism_pk)
    exp_type = models.ExperimentType.objects.get(pk=exp_type_pk)

    experiments = models.Experiment.objects.filter(
        dataset__assembly__organism=organism,
        experiment_type=exp_type,
    ).distinct()
    similarities = models.Similarity.objects.filter(
        experiment_1__in=experiments,
        experiment_2__in=experiments)

    network = Network(experiments, similarities)
    network_json = network.create_network_json()

    models.OrganismNetwork.objects.update_or_create(
        organism=organism,
        experiment_type=exp_type,
        defaults={
            'network_plot': network_json,
        },
    )
