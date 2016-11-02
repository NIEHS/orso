#!/usr/bin/env python

import pyBigWig
import os
import bisect
import json
import numpy
import click

from collections import defaultdict


def check_bigwigs(single_bw, paired_1_bw, paired_2_bw):
    if not single_bw and not paired_1_bw and not paired_2_bw:
        raise ValueError('No bigWigs specified.')
    if single_bw and (paired_1_bw or paired_2_bw):
        raise ValueError('A single bigWig of ambiguous strand may not be run with paired bigWigs.')  # noqa
    if not single_bw and not (paired_1_bw and paired_2_bw):
        raise ValueError('If a single bigWig of ambiguous strand is not used, paired bigWigs are required.')  # noqa

    if single_bw:
        try:
            pyBigWig.open(single_bw)
        except RuntimeError:
            print('pyBigWig is unable to open bigWig file.')
    else:
        try:
            pyBigWig.open(paired_1_bw)
            pyBigWig.open(paired_2_bw)
        except RuntimeError:
            print('pyBigWig is unable to open bigWig files.')


def find_intersection(start, end, intervals, interval_starts, interval_ends):
    index_1 = bisect.bisect_right(interval_starts, start)
    index_2 = bisect.bisect_left(interval_ends, end)

    intersecting_intervals = intervals[index_1-1:index_2+1]
    return intersecting_intervals


def find_bin_values(bin_start, bin_num, bin_size, intervals, interval_starts,
                    interval_ends):

    bin_values = []

    for i in range(bin_num):
        start = bin_start + i * bin_size
        end = start + bin_size - 1

        bin_range = (start, end)

        overlapping_intervals = find_intersection(
            start, end, intervals, interval_starts, interval_ends
        )

        coverage = 0
        for overlapping_interval in overlapping_intervals:
            val = min(bin_range[1], overlapping_interval[1]) - \
                    max(bin_range[0], overlapping_interval[0]) + 1
            positions = max(0, val)
            coverage += positions * overlapping_interval[2]
        bin_values.append(coverage)

    return bin_values


def get_starts_and_ends(intervals):
    starts = [r[0] for r in intervals]
    ends = [r[1] for r in intervals]
    return starts, ends


def get_bw_intervals(bw, chromosome):
    intervals = bw.intervals(chromosome)

    # Convert from 0-based to 1-based
    converted_intervals = []
    for interval in intervals:
        converted_intervals.append((interval[0]+1, interval[1], interval[2]))
    converted_intervals = tuple(converted_intervals)

    starts, ends = get_starts_and_ends(converted_intervals)

    return converted_intervals, starts, ends


def read_bed_by_chrom(bed_fn):
    chrom_entries = defaultdict(list)

    with open(bed_fn) as f:

        for line in f:
            line_split = line.strip().split('\t')
            chromosome = line_split[0]

            chrom_entries[chromosome].append(line_split)

    return chrom_entries


class MetaPlot(object):

    def __init__(self, bed_fn, bin_start, bin_num, bin_size, single_bw=None,
                 paired_1_bw=None, paired_2_bw=None):

        self.bed_fn = bed_fn
        self.bin_start = bin_start
        self.bin_num = bin_num
        self.bin_size = bin_size

        self.single_bw = single_bw
        self.paired_1_bw = paired_1_bw
        self.paired_2_bw = paired_2_bw

        assert os.path.exists(self.bed_fn)
        assert isinstance(self.bin_start, int)
        assert isinstance(self.bin_num, int)
        assert isinstance(self.bin_size, int)

        check_bigwigs(self.single_bw, self.paired_1_bw, self.paired_2_bw)

        self.find_data_matrix()

    def find_data_matrix(self):

        chrom_entries = read_bed_by_chrom(self.bed_fn)
        data_dict = defaultdict(list)
        self.data_matrix = {'matrix_rows': [], 'matrix_columns': []}

        if self.single_bw:
            bw = pyBigWig.open(self.single_bw)
        else:
            bw_1 = pyBigWig.open(self.paired_1_bw)
            bw_2 = pyBigWig.open(self.paired_2_bw)

        for chromosome, bed_values in chrom_entries.items():

            if self.single_bw:
                intervals, interval_starts, interval_ends = \
                    get_bw_intervals(bw, chromosome)
            else:
                intervals_1, interval_starts_1, interval_ends_1 = \
                    get_bw_intervals(bw_1, chromosome)
                intervals_2, interval_starts_2, interval_ends_2 = \
                    get_bw_intervals(bw_2, chromosome)

            for entry in bed_values:

                chromosome, start, end, name = entry[:4]
                if len(entry) > 4:
                    strand = entry[4]
                else:
                    strand = '.'
                center = int((int(end)-int(start))/2 + int(start))

                if strand == '+' or strand == '.':
                    intersection_start = center + self.bin_start
                    intersection_end = center + self.bin_start + \
                        self.bin_num * self.bin_size
                else:
                    intersection_start = center - self.bin_start - \
                        self.bin_num * self.bin_size
                    intersection_end = center - self.bin_start

                if self.single_bw:
                    intersected_intervals = find_intersection(
                        intersection_start, intersection_end, intervals,
                        interval_starts, interval_ends)
                    intersected_starts, intersected_ends = \
                        get_starts_and_ends(intersected_intervals)
                else:
                    if strand == '+':
                        intersected_intervals = find_intersection(
                            intersection_start, intersection_end, intervals_1,
                            interval_starts_1, interval_ends_1)
                        intersected_starts, intersected_ends = \
                            get_starts_and_ends(intersected_intervals)
                    if strand == '-':
                        intersected_intervals = find_intersection(
                            intersection_start, intersection_end, intervals_2,
                            interval_starts_2, interval_ends_2)
                        intersected_starts, intersected_ends = \
                            get_starts_and_ends(intersected_intervals)
                    if strand == '.':
                        intersected_intervals_1 = find_intersection(
                            intersection_start, intersection_end, intervals_1,
                            interval_starts_1, interval_ends_1)
                        intersected_starts_1, intersected_ends_1 = \
                            get_starts_and_ends(intersected_intervals_1)

                        intersected_intervals_2 = find_intersection(
                            intersection_start, intersection_end, intervals_2,
                            interval_starts_2, interval_ends_2)
                        intersected_starts_2, intersected_ends_2 = \
                            get_starts_and_ends(intersected_intervals_2)

                if strand == '+' or strand == '-':
                    bin_values = find_bin_values(
                        intersection_start,
                        self.bin_num,
                        self.bin_size,
                        intersected_intervals,
                        intersected_starts,
                        intersected_ends)
                    if strand == '-':
                        bin_values.reverse()

                if strand == '.':
                    if self.single_bw:
                        bin_values = find_bin_values(
                            intersection_start,
                            self.bin_num,
                            self.bin_size,
                            intersected_intervals,
                            intersected_starts,
                            intersected_ends)
                    else:
                        bin_values_1 = find_bin_values(
                            intersection_start,
                            self.bin_num,
                            self.bin_size,
                            intersected_intervals_1,
                            intersected_starts_1,
                            intersected_ends_1)
                        bin_values_2 = find_bin_values(
                            intersection_start,
                            self.bin_num,
                            self.bin_size,
                            intersected_intervals_2,
                            intersected_starts_2,
                            intersected_ends_2)
                        bin_values = \
                            [x + y for x, y in zip(bin_values_1, bin_values_2)]

                data_dict[name] = bin_values

        with open(self.bed_fn) as f:
            for line in f:
                name = line.strip().split()[3]
                self.data_matrix['matrix_rows'].append({
                    'name': name,
                    'row_values': data_dict[name]
                })

        for i in range(self.bin_num):
            s = self.bin_start + i * self.bin_size
            e = s + self.bin_size - 1

            self.data_matrix['matrix_columns'].append((s, e))

    def create_intersection_json(self, json_file=None):
        intersection_values = []

        for row in self.data_matrix['matrix_rows']:
            intersection_values.append(sum(row['row_values']))

        if json_file:
            with open(json_file, 'w') as outfile:
                json.dump(intersection_values, outfile)
        else:
            return json.dumps(intersection_values)

    def create_metaplot_json(self, json_file=None):
        matrix_values = []

        for row in self.data_matrix['matrix_rows']:
            matrix_values.append(row['row_values'])

        metaplot = {
            'metaplot_values':
                list(numpy.mean(matrix_values[0:2], axis=0)),
            'bin_values': self.data_matrix['matrix_columns'],
        }

        if json_file:
            with open(json_file, 'w') as outfile:
                json.dump(metaplot, outfile)
        else:
            return json.dumps(metaplot)


@click.command()
@click.argument('feature_bed')
@click.argument('output_header')
@click.option('-s', help='Single bigWig (Ambiguous strand)', type=str)
@click.option('-f', help='Forward strand (+) bigWig', type=str)
@click.option('-r', help='Reverse strand (-) bigWig', type=str)
@click.option('--bin_start', default=-2500, help='Relative bin start',
              type=int)
@click.option('--bin_number', default=50, help='Number of bins',
              type=int)
@click.option('--bin_size', default=100, help='Size of bins',
              type=int)
def cli(feature_bed, output_header, s, f, r, bin_start, bin_number, bin_size):
    """
    Generate a metaplot and intersection values for bigWig file(s) over BED \
    file entries.
    """

    metaplot = MetaPlot(
        feature_bed,
        bin_start,
        bin_number,
        bin_size,
        single_bw=s,
        paired_1_bw=f,
        paired_2_bw=r,
    )

    metaplot.create_intersection_json(output_header + '.intersection.json')
    metaplot.create_metaplot_json(output_header + '.metaplot.json')

if __name__ == '__main__':
    cli()
