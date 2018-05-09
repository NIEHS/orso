import json
import numpy as np
from celery import group
from celery.decorators import task
from scipy.cluster.hierarchy import dendrogram, linkage

from network import models
from network.tasks.utils import blend_colors, get_exp_color, hex_to_rgba


def cluster(experiments, similarities):
    edges = set()
    for sim in similarities:
        edges.add(tuple(sorted([
            sim.experiment_1.pk,
            sim.experiment_2.pk,
        ])))

    edge_matrix = np.zeros((len(experiments), len(experiments)), dtype=int)

    for i, exp_1 in enumerate(experiments):
        for j, exp_2 in enumerate(experiments):
            if exp_1 == exp_2:
                edge_matrix[i][j] = 1
            elif tuple(sorted([exp_1.pk, exp_2.pk])) in edges:
                edge_matrix[i][j] = 1

    link = linkage(edge_matrix, 'ward')
    dend = dendrogram(link)

    return link, dend


def get_all_children(node, node_dict):
    _all = set()
    for child in node_dict[node]['children']:
        _all.add(child)
        _all |= get_all_children(child, node_dict)
    return _all


def get_x_position(node, node_dict):
    if node_dict[node]['leaf']:
        return node_dict[node]['x']
    else:
        child_1, child_2 = list(node_dict[node]['children'])
        return np.mean([
            get_x_position(child_1, node_dict),
            get_x_position(child_2, node_dict),
        ])


def create_and_annotate_nodes(experiments, link, dend):
    leaf_order = dend['ivl']
    leaf_order = list([int(leaf) for leaf in leaf_order])

    nodes = dict()

    # Create nodes
    for i, index in enumerate(leaf_order):
        nodes[index] = {
            'children': set(),
            'leaf': True,
            'dist': 0,
            'x': i,
            'y': 0,
        }
    for i, row in enumerate(link):
        index = i + len(leaf_order)
        nodes[index] = {
            'children': set([int(row[0]), int(row[1])]),
            'leaf': False,
            'y': 1 + row[2],
        }

    # Annotate
    for index, node in nodes.items():
        if node['leaf']:
            node['color'] = hex_to_rgba(get_exp_color(experiments[index]))
            node['name'] = experiments[index].name
        else:
            children = get_all_children(index, nodes)
            child_leaves = [
                child for child in children if nodes[child]['leaf']]
            colors = [
                hex_to_rgba(get_exp_color(experiments[i]))
                for i in child_leaves]
            node['color'] = blend_colors(colors)
            node['x'] = get_x_position(index, nodes)

    return nodes


def create_dendrogram_plot(nodes):
    # For Plotly
    dots = []
    lines = []

    for index, node in nodes.items():
        if node['leaf']:
            dots.append({
                'x': node['x'],
                'y': node['y'],
                'color': node['color'],
                'text': node['name'],
            })
        else:
            child_index_1, child_index_2 = list(nodes[index]['children'])
            child_1 = nodes[child_index_1]
            child_2 = nodes[child_index_2]

            for child in [child_1, child_2]:
                lines.append({
                    'type': 'line',
                    'x0': node['x'],
                    'y0': node['y'],
                    'x1': child['x'],
                    'y1': node['y'],
                    'line': {
                        'color': 'rgba({}, {}, {}, {})'.format(
                            *child['color']),
                        'width': 3,
                    },
                })
                lines.append({
                    'type': 'line',
                    'x0': child['x'],
                    'y0': node['y'],
                    'x1': child['x'],
                    'y1': child['y'],
                    'line': {
                        'color': 'rgba({}, {}, {}, {})'.format(
                            *child['color']),
                        'width': 3,
                    },
                })

    data = [{
        'x': list([dot['x'] for dot in dots]),
        'y': list([dot['y'] for dot in dots]),
        'mode': 'markers',
        'marker': {
            'size': 16,
            'color': list(['rgba({}, {}, {}, {})'.format(
                *dot['color']) for dot in dots]),
        },
        'text': list([dot['text'] for dot in dots]),
    }]
    layout = {
        'shapes': lines,
    }

    return {
        'data': data,
        'layout': layout,
    }


def update_dendrogram(organism_pk, exp_type_pk):

    organism = models.Organism.objects.get(pk=organism_pk)
    exp_type = models.ExperimentType.objects.get(pk=exp_type_pk)

    experiments = models.Experiment.objects.filter(
        dataset__assembly__organism=organism,
        experiment_type=exp_type,
    ).distinct()
    similarities = models.Similarity.objects.filter(
        experiment_1__in=experiments,
        experiment_2__in=experiments)

    try:
        link, dend = cluster(experiments, similarities)
    except ValueError:
        print('Error during clustering, likely empty distance matrix; '
              '{} total experiments'.format(str(experiments.count())))
        raise
    else:
        nodes = create_and_annotate_nodes(experiments, link, dend)
        dendrogram_plot = create_dendrogram_plot(nodes)

        models.Dendrogram.objects.update_or_create(
            organism=organism,
            experiment_type=exp_type,
            defaults={
                'dendrogram_plot': json.dumps(dendrogram_plot),
            },
        )
