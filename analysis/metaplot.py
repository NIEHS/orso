#!/usr/bin/env python

import json
import math
from tempfile import NamedTemporaryFile

from analysis.utils import call_bigwig_average_over_bed

START = -2500
SIZE = 100
NUM = 50


def get_metaplot_values(gr_object, bed_path=None, ambiguous_bigwig=None,
                        plus_bigwig=None, minus_bigwig=None, bin_start=START,
                        bin_size=SIZE, bin_num=NUM):
    '''
    For GenomicRegions object, create metaplot and intersection values
    considering input BigWig files.

    bed_path - Path to BED file with bin information. See
        generate_metaplot_bed.
    '''
    # Generate BED file for intersection
    if not bed_path:
        metaplot_bed = NamedTemporaryFile(mode='w')
        generate_metaplot_bed(
            gr_object,
            json.loads(gr_object.assembly.chromosome_sizes),
            metaplot_bed,
            bin_start,
            bin_size,
            bin_num,
        )
        metaplot_bed.flush()
        bed_path = metaplot_bed.name

    entries = read_bed(gr_object.bed_file)

    # Run intersection, process output
    if plus_bigwig and minus_bigwig:
        plus_tab = NamedTemporaryFile(mode='w')
        minus_tab = NamedTemporaryFile(mode='w')

        call_bigwig_average_over_bed(
            plus_bigwig,
            bed_path,
            plus_tab.name,
        )
        call_bigwig_average_over_bed(
            minus_bigwig,
            bed_path,
            minus_tab.name,
        )

        plus_tab.flush()
        minus_tab.flush()

        metaplot_values = reconcile_metaplot_stranded_coverage(
            entries,
            read_metaplot_bigwig_average_over_bed(
                open(plus_tab.name), entries, bin_num),
            read_metaplot_bigwig_average_over_bed(
                open(minus_tab.name), entries, bin_num),
        )

        return(
            read_metaplot_from_values(
                metaplot_values, bin_start, bin_size, bin_num),
            read_intersection_from_values(
                metaplot_values, gr_object, bin_num),
        )

    elif ambiguous_bigwig:
        tab = NamedTemporaryFile(mode='w')

        call_bigwig_average_over_bed(
            ambiguous_bigwig,
            bed_path,
            tab.name,
        )

        tab.flush()

        metaplot_values = read_metaplot_bigwig_average_over_bed(
            open(tab.name), entries, bin_num)

        return(
            read_metaplot_from_values(
                metaplot_values, bin_start, bin_size, bin_num),
            read_intersection_from_values(
                metaplot_values, gr_object, bin_num),
        )

    else:
        raise ValueError('Improper bigWig files specified.')


def read_metaplot_from_values(entry_dict, bin_start, bin_size, bin_num):
    '''
    From a dictionary describing binned coverage values for each entry, create
    a dictionary describing the associated metaplot.
    '''
    metaplot_values = [0] * bin_num
    for entry_values in entry_dict.values():
        for i, value in enumerate(entry_values):
            metaplot_values[i] += value
    entry_num = len(entry_dict)
    for i, value in enumerate(metaplot_values):
        metaplot_values[i] = value / entry_num

    bin_values = []
    for i in range(bin_num):
        bin_values.append([
            bin_start + i * bin_size,
            bin_start + (i + 1) * bin_size - 1,
        ])

    return {
        'metaplot_values': metaplot_values,
        'bin_values': bin_values,
    }


def read_intersection_from_values(entry_dict, gr_object, bin_num):
    '''
    From a dictionary describing binned coverage values for each entry, create
    a list describing average coverage for each entry, ordered by the
    GenomicRegions BED file.
    '''
    intersection_list = []
    for line in [_line.decode('utf-8') for _line in gr_object.bed_file]:
        if not is_header(line):
            entry_name = line.strip().split()[3]
            intersection_list.append(sum(entry_dict[entry_name]) / bin_num)
    return intersection_list


def read_metaplot_bigwig_average_over_bed(tab_file, entries, bin_num):
    '''
    Read metaplot tab file. Return metaplot values.
    '''
    metaplot_values = dict()
    for entry in entries:
        metaplot_values[entry['name']] = [0] * bin_num

    for line in tab_file:
        name, size, covered, value_sum, mean, mean0 = line.strip().split()
        entry_name = '_'.join(name.split('_')[:-1])
        index = int(name.split('_')[-1])
        metaplot_values[entry_name][index] = float(value_sum)

    return metaplot_values


def reconcile_metaplot_stranded_coverage(entries, plus_values, minus_values):
    '''
    Considering plus and minus metaplot values, return only coverage values of
    the appropriate strand.
    '''
    values = dict()
    for entry in entries:
        name, strand = (entry['name'], entry['strand'])
        if strand == '+':
            values[name] = plus_values[name]
        elif strand == '-':
            values[name] = minus_values[name]
        elif strand == '.':
            values[name] = \
                [x + y for x, y in zip(plus_values[name], minus_values[name])]
    return values


def generate_metaplot_bed(genomic_regions_obj, chrom_sizes_dict,
                          output_file_obj, bin_start=START, bin_size=SIZE,
                          bin_num=NUM):
    '''
    Write a BED file to output_file_obj containing entries for each genomic
    ranges entry.
    '''
    OUT = output_file_obj

    def write_bin_to_out(entry, _bin, index):
        '''
        Write bin to OUT in BED6 format.
        '''
        size = chrom_sizes_dict[entry['chromosome']]
        if _bin[0] < size and _bin[1] > 0:
            OUT.write('\t'.join([
                entry['chromosome'],
                str(max(0, _bin[0])),
                str(min(size, _bin[1])),
                '{}_{}'.format(entry['name'], str(index)),
                '0',
                entry['strand'],
            ]) + '\n')

    bed_entries = read_bed(genomic_regions_obj.bed_file)
    for entry in bed_entries:
        if entry['strand'] == '+' or entry['strand'] == '.':
            start = entry['center'] + bin_start
            for i in range(bin_num):
                _bin = [start + (i - 1) * bin_size, start + i * bin_size]
                write_bin_to_out(entry, _bin, i)
        elif entry['strand'] == '-':
            start = entry['center'] - bin_start
            for i in range(bin_num):
                _bin = [start - i * bin_size, start - (i - 1) * bin_size]
                write_bin_to_out(entry, _bin, i)

    OUT.flush()


def read_bed(bed_file):
    '''
    Return a list of entries for each line in a BED file.
    '''
    bed_entries = []
    for line in [_line.decode('utf-8') for _line in bed_file]:
        if not is_header(line):
            line_split = line.strip().split('\t')

            chromosome, start, end, name = line_split[:4]
            start = int(start)
            end = int(end)
            center = math.floor((start + end) / 2)
            if len(line_split) > 5:
                strand = line_split[5]
            else:
                strand = '.'

            bed_entries.append({
                'chromosome': chromosome,
                'start': start,
                'end': end,
                'center': center,
                'name': name,
                'strand': strand,
            })
    return bed_entries
