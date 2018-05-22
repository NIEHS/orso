import json
import random

import networkx as nx
import numpy
from celery import group
from celery.decorators import task
from django.db.models import Q
from fa2 import ForceAtlas2

from network import models
from network.tasks.utils import blend_colors, get_exp_color, hex_to_rgba, \
    rgba_to_string


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
                'size': 1,
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

        network = {
            'nodes': node_list,
            'edges': edge_list,
        }

        sub_graphs = nx.connected_component_subgraphs(g)

        max_nodes = 0
        max_node_subgraph = None

        for i, sg in enumerate(sub_graphs):
            if sg.number_of_nodes() > max_nodes:
                max_nodes = sg.number_of_nodes()
                max_node_subgraph = sg

        subgraph_nodes = max_node_subgraph.nodes()

        max_total_x = float('-inf')
        min_total_x = float('inf')
        max_total_y = float('-inf')
        min_total_y = float('inf')

        max_field_x = float('-inf')
        min_field_x = float('inf')
        max_field_y = float('-inf')
        min_field_y = float('inf')

        for node in network['nodes']:

            max_total_x = max(max_total_x, node['x'])
            min_total_x = min(min_total_x, node['x'])
            max_total_y = max(max_total_y, node['y'])
            min_total_y = min(min_total_y, node['y'])

            if node['id'] in subgraph_nodes:
                max_field_x = max(max_field_x, node['x'])
                min_field_x = min(min_field_x, node['x'])
                max_field_y = max(max_field_y, node['y'])
                min_field_y = min(min_field_y, node['y'])

            x_position = numpy.mean([max_field_x, min_field_x])
            y_position = numpy.mean([max_field_y, min_field_y])

        if max_nodes > 1:
            x_zoom = (max_field_x - min_field_x) / \
                (max_total_x - min_total_x)
            y_zoom = (max_field_y - min_field_y) / \
                (max_total_y - min_total_y)
            zoom_ratio = min(1.1 * max(x_zoom, y_zoom), 1)
        else:
            zoom_ratio = 1

        network['nodes'].append({
            'x': x_position,
            'y': y_position,
            'id': 'center',
            'color': rgba_to_string((0, 0, 0, 0)),
            'size': 1,
        })

        network['camera'] = {
            'zoom_ratio': zoom_ratio,
        }

        return json.dumps(network)


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

            experiments = models.Experiment.objects.filter(
                dataset__assembly__organism=org,
                experiment_type=exp_type,
            )

            if experiments.exists():
                tasks.append(
                    update_organism_network.si(org.pk, exp_type.pk))

                for my_user in models.MyUser.objects.all():
                    if experiments.filter(owners=my_user).exists():
                        tasks.append(update_organism_network.si(
                            org.pk, exp_type.pk, my_user_pk=my_user.pk))

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
def update_organism_network(organism_pk, exp_type_pk, my_user_pk=None):

    organism = models.Organism.objects.get(pk=organism_pk)
    exp_type = models.ExperimentType.objects.get(pk=exp_type_pk)

    experiment_query = (Q(dataset__assembly__organism=organism) &
                        Q(experiment_type=exp_type))

    if my_user_pk:
        my_user = models.MyUser.objects.get(pk=my_user_pk)
        experiment_query &= (Q(owners=None) | Q(owners=my_user))
    else:
        my_user = None
        experiment_query &= Q(owners=None)

    experiments = models.Experiment.objects.filter(experiment_query).distinct()
    similarities = models.Similarity.objects.filter(
        experiment_1__in=experiments,
        experiment_2__in=experiments)

    network = Network(experiments, similarities)
    network_json = network.create_network_json()

    models.OrganismNetwork.objects.update_or_create(
        organism=organism,
        experiment_type=exp_type,
        my_user=my_user,
        defaults={
            'network_plot': network_json,
        },
    )
