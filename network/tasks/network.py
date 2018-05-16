import json
import random

import networkx as nx
from celery import group
from celery.decorators import task
from django.db.models import Q
from fa2 import ForceAtlas2

from network import models
from network.tasks.utils import blend_colors, get_exp_color, hex_to_rgba


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

        for obj in self.node_objects_list:
            g.add_node(self.get_node_pk(obj))

        for obj_1 in self.node_objects_list:
            for obj_2 in self.node_objects_list:
                pk_1, pk_2 = tuple(sorted([
                    self.get_node_pk(obj_1),
                    self.get_node_pk(obj_2),
                ]))
                if any([
                    (pk_1, pk_2) in edges,
                    self.check_node_equivalence(obj_1, obj_2)
                ]):
                    g.add_edge(pk_1, pk_2)

        fa2 = ForceAtlas2()
        try:
            positions = fa2.forceatlas2_networkx_layout(
                g, pos=None, iterations=2000)
        except ZeroDivisionError:
            positions = dict()
            for obj in self.node_objects_list:
                positions[self.get_node_pk(obj)] = \
                    (random.random(), random.random())

        node_list = []

        for obj in self.node_objects_list:

            pk = self.get_node_pk(obj)
            position = positions[pk]

            hex_color = self.get_node_color(obj)
            rgba = hex_to_rgba(hex_color)

            node_list.append({
                'id': self.get_node_pk(obj),
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
            for obj_2 in self.node_objects_list[(i + 1):]:
                pk_1, pk_2 = tuple(sorted([
                    self.get_node_pk(obj_1),
                    self.get_node_pk(obj_2),
                ]))
                if any([
                    (pk_1, pk_2) in edges,
                    self.check_node_equivalence(obj_1, obj_2)
                ]):

                    hex_1 = self.get_node_color(obj_1)
                    rgba_1 = hex_to_rgba(hex_1)

                    hex_2 = self.get_node_color(obj_2)
                    rgba_2 = hex_to_rgba(hex_2)

                    rgba = blend_colors([rgba_1, rgba_2])

                    edge_list.append({
                        'id': edge_count,
                        'source': pk_1,
                        'target': pk_2,
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
