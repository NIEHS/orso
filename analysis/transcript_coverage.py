import json
import math
from collections import defaultdict
from tempfile import NamedTemporaryFile

from analysis.utils import call_bigwig_average_over_bed
from network import models


def get_locus_values(loci, locus_bed_path, ambiguous_bigwig=None,
                     plus_bigwig=None, minus_bigwig=None):
    '''
    Finds coverage values for each transcript.

    loci - Dict of locus objects from models.LocusGroup.get_loci_dict()
    locus_bed_bed - Path to BED file with loci intervals.
    '''
    if plus_bigwig and minus_bigwig:
        plus_tab = NamedTemporaryFile(mode='w')
        minus_tab = NamedTemporaryFile(mode='w')

        call_bigwig_average_over_bed(
            plus_bigwig,
            locus_bed_path,
            plus_tab.name,
        )
        call_bigwig_average_over_bed(
            minus_bigwig,
            locus_bed_path,
            minus_tab.name,
        )

        plus_tab.flush()
        minus_tab.flush()

        return reconcile_stranded_coverage(
            loci,
            read_bigwig_average_over_bed_tab_file(plus_tab.name),
            read_bigwig_average_over_bed_tab_file(minus_tab.name),
        )

    elif ambiguous_bigwig:
        tab = NamedTemporaryFile(mode='w')

        call_bigwig_average_over_bed(
            ambiguous_bigwig,
            locus_bed_path,
            tab.name,
        )
        tab.flush()

        out_values = read_bigwig_average_over_bed_tab_file(tab.name)
        return out_values

    else:
        raise ValueError('Improper bigWig files specified.')


def read_bigwig_average_over_bed_tab_file(tab_file_path):
    '''
    Read values in bigWigAverageOverBed output file into dict.
    '''
    locus_values = defaultdict(float)
    with open(tab_file_path) as f:
        for line in f:
            name, size, covered, value_sum, mean, mean0 = line.strip().split()
            locus_pk = int(name.split('_')[0])
            locus_values[locus_pk] += float(value_sum)
    return locus_values


def reconcile_stranded_coverage(loci, plus_values, minus_values):
    '''
    Considering plus and minus strand coverage values, return only coverage
    values of the appropriate strand.
    '''
    reconciled = dict()
    for locus in loci:
        if locus.strand is None:
            reconciled[locus.pk] = plus_values[locus.pk] + \
                minus_values[locus.pk]
        elif locus.strand == '+':
            reconciled[locus.pk] = plus_values[locus.pk]
        elif locus.strand == '-':
            reconciled[locus.pk] = minus_values[locus.pk]
    return reconciled


def generate_locusgroup_bed(locus_group, output_file_obj):
    '''
    Write a BED file to output_file_obj containing entries for each locus in a
    locus group.
    '''
    OUT = output_file_obj

    def write_to_out(locus, interval, index):
        '''
        Write interval to OUT in BED6 format
        '''
        if locus.strand:
            strand = locus.strand
        else:
            strand = '.'
        OUT.write('\t'.join([
            locus.chromosome,
            str(interval[0] - 1),
            str(interval[1]),
            '{}_{}'.format(str(locus.pk), str(index)),
            '0',
            strand,
        ]) + '\n')

    chrom_sizes = json.loads(locus_group.assembly.chromosome_sizes)

    for locus in models.Locus.objects.filter(group=locus_group):
        for i, region in enumerate(locus.regions):

            if locus_group.group_type in ['promoter', 'enhancer']:

                center = math.floor((region[0] + region[1]) / 2)
                if locus.strand == '+' or locus.strand is None:
                    interval = [
                        max(center - 2500, 1),
                        min(center + 2499, chrom_sizes[locus.chromosome]),
                    ]

                elif locus.strand == '-':
                    interval = [
                        max(center - 2499, 1),
                        min(center + 2500, chrom_sizes[locus.chromosome]),
                    ]

                write_to_out(locus, interval, i)

            elif locus_group.group_type in ['genebody', 'mRNA']:
                write_to_out(locus, region, i)

    OUT.flush()
