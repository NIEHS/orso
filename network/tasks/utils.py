import json
import math
import os

import numpy as np
from django.conf import settings
from django.core.cache import cache

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
