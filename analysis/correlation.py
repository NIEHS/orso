#!/usr/bin/env python

import os
import json
import click
import numpy
from scipy import stats


class Correlation(object):

    def __init__(self, intersection_json_1, intersection_json_2):

        self.intersection_1 = intersection_json_1
        self.intersection_2 = intersection_json_2

    def get_correlation(self):

        i1_vector = numpy.array(self.intersection_1)
        i2_vector = numpy.array(self.intersection_2)

        return stats.spearmanr(i1_vector, i2_vector)


@click.command()
@click.argument('intersection_file_1')
@click.argument('intersection_file_2')
def cli(intersection_json_1, intersection_json_2):
    """
    Find a correlation value considering paired intersection files.
    """

    correlation = Correlation(
        intersection_json_1,
        intersection_json_2,
    )

    print(correlation.get_correlation())

if __name__ == '__main__':
    cli()
