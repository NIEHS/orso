#!/usr/bin/env python

import os
import json
import click
import numpy
from scipy import stats


class Correlation(object):

    def __init__(self, intersection_file_1, intersection_file_2):

        assert os.path.exists(intersection_file_1)
        assert os.path.exists(intersection_file_2)

        with open(intersection_file_1) as f1, open(intersection_file_2) as f2:
            self.intersection_1 = json.load(f1)
            self.intersection_2 = json.load(f2)

    def get_correlation(self):

        i1_vector = numpy.array(self.intersection_1)
        i2_vector = numpy.array(self.intersection_2)

        return stats.spearmanr(i1_vector, i2_vector)


@click.command()
@click.argument('intersection_file_1')
@click.argument('intersection_file_2')
def cli(intersection_file_1, intersection_file_2):
    """
    Find a correlation value considering paired intersection files.
    """

    correlation = Correlation(
        intersection_file_1,
        intersection_file_2,
    )

    print(correlation.get_correlation())

if __name__ == '__main__':
    cli()
