import json
import random

import networkx as nx
import numpy
from celery import group
from celery.decorators import task
from django.db.models import Q
from fa2 import ForceAtlas2

from network import models
from network.tasks.utils import blend_colors, get_exp_color, get_exp_tag, \
    hex_to_rgba, rgba_to_string


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

    if experiments:

        similarities = models.Similarity.objects.filter(
            experiment_1__in=experiments,
            experiment_2__in=experiments)

        node_list = []
        exp_to_tag = dict()
        tag_to_node = dict()
        current_index = 0
        for experiment in experiments:
            exp_tag = get_exp_tag(experiment)
            exp_to_tag[experiment.pk] = exp_tag
            if exp_tag in tag_to_node:
                i = tag_to_node[exp_tag]
                node_list[i]['size'] += 1
            else:
                tag_to_node[exp_tag] = current_index
                rgba = get_exp_color(experiment)
                node_list.append({
                    'id': current_index,
                    'label': exp_tag,
                    'size': 1,
                    'color': 'rgba({}, {}, {}, {})'.format(
                        *[str(n) for n in rgba]),
                    'exp_tag': exp_tag,
                })
                current_index += 1

        edge_list = []
        nodes_to_edge = dict()
        current_index = 0
        for similarity in similarities:
            node_tuple = tuple(sorted([
                tag_to_node[exp_to_tag[similarity.experiment_1.pk]],
                tag_to_node[exp_to_tag[similarity.experiment_2.pk]],
            ]))
            if node_tuple[0] != node_tuple[1]:
                if node_tuple in nodes_to_edge:
                    i = nodes_to_edge[node_tuple]
                    edge_list[i]['size'] += 1
                else:
                    nodes_to_edge[node_tuple] = current_index
                    rgba = blend_colors([
                        get_exp_color(similarity.experiment_1),
                        get_exp_color(similarity.experiment_2),
                    ])
                    edge_list.append({
                        'id': current_index,
                        'size': 1,
                        'source': node_tuple[0],
                        'target': node_tuple[1],
                        'color': 'rgba({}, {}, {}, {})'.format(
                            *[str(n) for n in rgba]),
                    })
                    current_index += 1

        g = nx.Graph()

        for i in range(len(node_list)):
            g.add_node(i)

        for edge in edge_list:
            g.add_edge(edge['source'], edge['target'])

        fa2 = ForceAtlas2()
        try:
            positions = fa2.forceatlas2_networkx_layout(
                g, pos=None, iterations=50)
        except ZeroDivisionError:
            positions = dict()
            for i in range(len(node_list)):
                positions[i] = (random.random(), random.random())

        for i in range(len(node_list)):
            node_list[i]['x'] = positions[i][0]
            node_list[i]['y'] = positions[i][1]

        models.OrganismNetwork.objects.update_or_create(
            organism=organism,
            experiment_type=exp_type,
            my_user=my_user,
            defaults={
                'network_plot': json.dumps({
                    'nodes': node_list,
                    'edges': edge_list,
                }),
            },
        )
