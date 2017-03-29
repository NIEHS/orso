#!/usr/bin/env python

import pyBigWig
import os
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
            return False
    else:
        try:
            pyBigWig.open(paired_1_bw)
            pyBigWig.open(paired_2_bw)
        except RuntimeError:
            print('pyBigWig is unable to open bigWig files.')
            return False

    return True


def is_header(line):
    line_split = line.strip().split('\t')
    if 'track' in line_split[0] or 'browser' in line_split[0]:
        return True


def check_position(value, max_length):
    if value < 1:
        return 1
    elif value > max_length:
        return max_length
    else:
        return value


def find_bin_values(bin_start, bin_num, bin_size, intervals):

    bin_values = []

    if bin_start:
        for i in range(bin_num):
            start = bin_start + i * bin_size
            end = start + bin_size - 1

            bin_range = (start, end)

            coverage = 0
            if intervals:
                for interval in intervals:
                    val = min(bin_range[1], interval[1]) - \
                        max(bin_range[0], interval[0]) + 1
                    positions = max(0, val)
                    coverage += positions * interval[2]
            bin_values.append(coverage / bin_size)
    else:
        for i in range(bin_num):
            bin_values.append(0)

    return bin_values


def convert_intervals(intervals):
    converted_intervals = []
    if intervals:
        for interval in intervals:
            converted_intervals.append(
                (interval[0] + 1, interval[1], interval[2]))
        converted_intervals = tuple(converted_intervals)
        return converted_intervals
    else:
        return intervals


def read_bed(bed_fn):
    bed_entries = []
    with open(bed_fn) as f:
        for line in f:
            line_split = line.strip().split('\t')
            if not is_header(line):
                bed_entries.append(line_split)
    return bed_entries


class MetaPlot(object):

    def __init__(self, bed_fn, bin_start=-2500, bin_num=50, bin_size=100,
                 single_bw=None, paired_1_bw=None, paired_2_bw=None):

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

        if check_bigwigs(self.single_bw, self.paired_1_bw, self.paired_2_bw):
            self.find_data_matrix()

    def find_data_matrix(self):

        # chrom_entries = read_bed_by_chrom(self.bed_fn)
        bed_entries = read_bed(self.bed_fn)
        data_dict = defaultdict(list)
        self.data_matrix = {'matrix_rows': [], 'matrix_columns': []}

        if self.single_bw:
            bw = pyBigWig.open(self.single_bw)
            bw_chroms = set(bw.chroms().keys())
            chrom_sizes = bw.chroms()
        else:
            bw_1 = pyBigWig.open(self.paired_1_bw)
            bw_2 = pyBigWig.open(self.paired_2_bw)
            bw_chroms = set(bw_1.chroms().keys())
            for chrom in bw_2.chroms().keys():
                bw_chroms.add(chrom)
            chrom_sizes = bw_1.chroms()
            for key, value in bw_2.chroms().items():
                if key in chrom_sizes:
                    chrom_sizes[key] = max([chrom_sizes[key], value])
                else:
                    chrom_sizes[key] = value

        for entry in bed_entries:

            chromosome, start, end, name = entry[:4]
            if len(entry) > 5:
                strand = entry[5]
            else:
                strand = '.'
            center = \
                int((int(end) - int(start) + 1) / 2 + (int(start) + 1))

            if strand == '+' or strand == '.':
                intersection_start = center + self.bin_start
                intersection_end = center + self.bin_start + \
                    self.bin_num * self.bin_size
            else:
                intersection_start = center - self.bin_start - \
                    self.bin_num * self.bin_size
                intersection_end = center - self.bin_start

            try:
                intersection_start = \
                    check_position(intersection_start, chrom_sizes[chromosome])
                intersection_end = \
                    check_position(intersection_end, chrom_sizes[chromosome])
            except KeyError:  # if chromosome not in chrom_sizes
                intersection_start = None
                intersection_end = None

            if self.single_bw:
                try:
                    intervals = convert_intervals(bw.intervals(
                        chromosome, intersection_start - 1, intersection_end))
                except:
                    intervals = ()
            else:
                if strand == '+':
                    try:
                        intervals = convert_intervals(bw_1.intervals(
                            chromosome,
                            intersection_start - 1,
                            intersection_end))
                    except:
                        intervals = ()
                if strand == '-':
                    try:
                        intervals = convert_intervals(bw_2.intervals(
                            chromosome,
                            intersection_start - 1,
                            intersection_end))
                    except:
                        intervals = ()
                if strand == '.':
                    try:
                        intervals_1 = convert_intervals(bw_1.intervals(
                            chromosome,
                            intersection_start - 1,
                            intersection_end))
                    except:
                        intervals_1 = ()
                    try:
                        intervals_2 = convert_intervals(bw_2.intervals(
                            chromosome,
                            intersection_start - 1,
                            intersection_end))
                    except:
                        intervals_2 = ()

            if strand == '+' or strand == '-':
                bin_values = find_bin_values(
                    intersection_start,
                    self.bin_num,
                    self.bin_size,
                    intervals)
                if strand == '-':
                    bin_values.reverse()

            if strand == '.':
                if self.single_bw:
                    bin_values = find_bin_values(
                        intersection_start,
                        self.bin_num,
                        self.bin_size,
                        intervals)
                else:
                    bin_values_1 = find_bin_values(
                        intersection_start,
                        self.bin_num,
                        self.bin_size,
                        intervals_1)
                    bin_values_2 = find_bin_values(
                        intersection_start,
                        self.bin_num,
                        self.bin_size,
                        intervals_2)
                    bin_values = \
                        [x + y for x, y in zip(bin_values_1,
                                               bin_values_2)]

            data_dict[name] = bin_values

        with open(self.bed_fn) as f:
            for line in f:
                if not is_header(line):
                    chrom = line.strip().split()[0]
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
            return intersection_values

    def create_metaplot_json(self, json_file=None):
        matrix_values = []

        for row in self.data_matrix['matrix_rows']:
            matrix_values.append(row['row_values'])
        metaplot = {
            'metaplot_values':
                list(numpy.mean(matrix_values, axis=0)),
            'bin_values': self.data_matrix['matrix_columns'],
        }

        if json_file:
            with open(json_file, 'w') as outfile:
                json.dump(metaplot, outfile)
        else:
            return metaplot


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

    try:
        metaplot.create_intersection_json(output_header + '.intersection.json')
        metaplot.create_metaplot_json(output_header + '.metaplot.json')
    except AttributeError:
        print('Unable to create output JSONs: {}'.format(str(vars(metaplot))))

if __name__ == '__main__':
    cli()
