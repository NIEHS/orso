import json
import math
import os

import numpy as np
import parse
from django.conf import settings
from django.core.cache import cache

TARGET_RELEVANT_EXP_TYPES = [
    'ChIP-seq',
    'siRNA knockdown followed by RNA-seq',
    'shRNA knockdown followed by RNA-seq',
    'CRISPR genome editing followed by RNA-seq',
    'CRISPRi followed by RNA-seq',
]
RGBA_STRING = 'rgba({}, {}, {}, {})'


def string_to_rgba(string):
    return list(parse.parse(RGBA_STRING, string))


def rgba_to_string(rgba):
    return RGBA_STRING.format(*[str(n) for n in rgba])


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


def blend_colors(colors):

    def _blend_color(colors):
        total = 0
        for color in colors:
            total += (1 / len(colors)) * math.pow(color, 2.2)
        return math.pow(total, 1 / 2.2)

    reds = [rgba[0] for rgba in colors]
    greens = [rgba[1] for rgba in colors]
    blues = [rgba[2] for rgba in colors]
    alphas = [rgba[3] for rgba in colors]

    red = _blend_color(reds)
    green = _blend_color(greens)
    blue = _blend_color(blues)

    alpha = np.mean(alphas)

    return (red, green, blue, alpha)


def retrieve_color_guide(target_color_guide, json_path):
    guide = cache.get(target_color_guide, None)
    if guide:
        return guide
    else:
        with open(json_path) as f:
            guide = json.load(f)
        cache.set(target_color_guide, guide)
        return guide


def get_target_color(target):
    target_color_guide = 'target_color_guide'
    json_path = os.path.join(settings.COLOR_KEY_DIR, 'target.json')
    color_guide = retrieve_color_guide(target_color_guide, json_path)
    try:
        color = color_guide[target]
    except KeyError:
        color = '#A9A9A9'
    return color


def get_cell_type_color(cell_type):
    target_color_guide = 'cell_type_color_guide'
    json_path = os.path.join(settings.COLOR_KEY_DIR, 'cell_type.json')
    color_guide = retrieve_color_guide(target_color_guide, json_path)

    try:
        color = color_guide[cell_type]
    except KeyError:
        color = '#A9A9A9'
    return color


def get_exp_color(experiment):

    color = None
    priority_list = ['target', 'cell_type']

    for attr in priority_list:

        value = getattr(experiment, attr)

        if value:
            if attr == 'cell_type':
                json_path = os.path.join(
                    settings.COLOR_KEY_DIR, 'cell_type.json')
                color_guide = retrieve_color_guide(
                    'cell_type_color_guide', json_path)
            elif attr == 'target':
                json_path = os.path.join(
                    settings.COLOR_KEY_DIR, 'target.json')
                color_guide = retrieve_color_guide(
                    'target_color_guide', json_path)

            try:
                color = color_guide[value]
            except KeyError:
                pass
            else:
                break

    if color:
        return hex_to_rgba(color)
    else:
        return hex_to_rgba('#A9A9A9')


def get_exp_tag(experiment):
    tag_list = []

    for attr in ['cell_type', 'target']:
        value = getattr(experiment, attr)
        if value:
            tag_list.append(value)

    return ', '.join(tag_list)
