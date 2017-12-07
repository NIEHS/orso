import copy
import json
import math
from tempfile import NamedTemporaryFile

from analysis.utils import call_bigwig_average_over_bed
from network import models

# BIN_VALUES give the associated bins for metaplot values
_values = ['{}:{}'.format(str(x), str(y)) for x, y in zip(
    range(-2500, 2500, 100), range(-2400, 2600, 100))]
BIN_VALUES = {
    'promoter': copy.deepcopy(_values),
    'enhancer': copy.deepcopy(_values),
}

_values = ['{}:{}'.format(str(x), str(y)) for x, y in zip(
    range(-2500, 0, 250), range(-2250, 250, 250))]
_values.extend([
    'genebody_{}:genebody_{}'.format(str(i), str(i + 1)) for i in range(30)])
_values.extend(['{}:{}'.format(str(x), str(y)) for x, y in zip(
    range(0, 2500, 250), range(250, 2750, 250))])
BIN_VALUES.update({
    'genebody': copy.deepcopy(_values),
    'mRNA': copy.deepcopy(_values),
})

# TICKS give the tick text for use in Plot.ly JS
_ticks = {
    'tickvals': ['-2000:-1900', '-1000:-900', '0:100', '1000:1100',
                 '2000:2100'],
    'ticktext': ['-2000', '-1000', '0', '1000', '2000'],
}
TICKS = {
    'promoter': copy.deepcopy(_ticks),
    'enhancer': copy.deepcopy(_ticks),
}

_ticks = {
    'tickvals': ['-2000:-1750', 'genebody_0:genebody_1',
                 'genebody_29:genebody_30', '2000:2250'],
    'ticktext': ['-2 kb', 'Start', 'End', '+2 kb'],
}
TICKS.update({
    'genebody': copy.deepcopy(_ticks),
    'mRNA': copy.deepcopy(_ticks),
})


def get_metaplot_values(lg_object, bed_path=None, ambiguous_bigwig=None,
                        plus_bigwig=None, minus_bigwig=None):
    '''
    For LocusGroup object, create metaplot and intersection values
    considering input BigWig files.

    bed_path - Path to BED file with bin information. See
        generate_metaplot_bed.
    '''
    # Generate BED file for intersection
    if not bed_path:
        metaplot_bed = NamedTemporaryFile(mode='w')
        generate_metaplot_bed(
            lg_object,
            json.loads(lg_object.assembly.chromosome_sizes),
            metaplot_bed,
        )
        metaplot_bed.flush()
        bed_path = metaplot_bed.name

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
            read_metaplot_bigwig_average_over_bed(open(plus_tab.name)),
            read_metaplot_bigwig_average_over_bed(open(minus_tab.name)),
        )

    elif ambiguous_bigwig:
        tab = NamedTemporaryFile(mode='w')

        call_bigwig_average_over_bed(
            ambiguous_bigwig,
            bed_path,
            tab.name,
        )

        tab.flush()

        metaplot_values = read_metaplot_bigwig_average_over_bed(open(tab.name))

    else:
        raise ValueError('Improper bigWig files specified.')

    return {
        'metaplot_values': metaplot_values,
        'bin_values': BIN_VALUES[lg_object.group_type],
        'ticks': TICKS[lg_object.group_type],
    }


def read_metaplot_bigwig_average_over_bed(tab_file):
    '''
    Read metaplot tab file. Return a list of average metaplot values.
    '''
    metaplot_values = [0] * 50
    entry_name_set = set()

    for line in tab_file:
        name, size, covered, value_sum, mean, mean0 = line.strip().split()

        entry_name = '_'.join(name.split('_')[:-1])
        entry_name_set.add(entry_name)

        index = int(name.split('_')[-1])
        metaplot_values[index] += float(mean0)

    entry_num = len(entry_name_set)
    for i, value in enumerate(metaplot_values):
        try:
            metaplot_values[i] = value / entry_num
        except ZeroDivisionError:
            metaplot_values[i] = 0

    return metaplot_values


def reconcile_metaplot_stranded_coverage(plus_values, minus_values):
    '''
    Considering plus and minus metaplot values, return only coverage values of
    the appropriate strand.
    '''
    values = []
    for plus, minus in zip(plus_values, minus_values):
        values.append((plus + minus) / 2)
    return values


def generate_metaplot_bed(locus_group_obj, output_file_obj):
    '''
    Write a BED file to output_file_obj containing entries for each locus in a
    group.
    '''
    OUT = output_file_obj
    chrom_sizes_dict = json.loads(locus_group_obj.assembly.chromosome_sizes)

    def write_bin_to_out(locus, _bin, index):
        '''
        Write bin to OUT in BED6 format.
        '''
        size = chrom_sizes_dict[locus.chromosome]
        if locus.strand:
            strand = locus.strand
        else:
            strand = '.'

        if _bin[0] < size and _bin[1] > 0:
            OUT.write('\t'.join([
                locus.chromosome,
                str(max(0, _bin[0])),
                str(min(size, _bin[1])),
                '{}_{}'.format(str(locus.pk), str(index)),
                '0',
                strand,
            ]) + '\n')

    loci = models.Locus.objects.filter(group=locus_group_obj)

    for locus in loci:
        locus_start = locus.regions[0][0]
        locus_end = locus.regions[-1][1]

        # Make 30 bins across the length of the genebody; 250-bp bins extending
        # to either side of the gene
        if locus_group_obj.group_type in ['genebody', 'mRNA']:
            index = 0

            if locus.strand == '+' or locus.strand is None:

                start = locus_start - 2500
                for i in range(10):
                    _bin = [start + i * 250, start + (i + 1) * 250]
                    write_bin_to_out(locus, _bin, index)
                    index += 1

                for i in range(30):
                    _bin = [
                        locus_start + math.floor(
                            (locus_end - locus_start) * (i / 30)),
                        locus_start + math.floor(
                            (locus_end - locus_start) * ((i + 1) / 30)),
                    ]
                    write_bin_to_out(locus, _bin, index)
                    index += 1

                start = locus_end
                for i in range(10):
                    _bin = [start + i * 250, start + (i + 1) * 250]
                    write_bin_to_out(locus, _bin, index)
                    index += 1

            elif locus.strand == '-':

                start = locus_end + 2500
                for i in range(10):
                    _bin = [start - (i + 1) * 250, start - i * 250]
                    write_bin_to_out(locus, _bin, index)
                    index += 1

                for i in range(30):
                    _bin = [
                        locus_end - math.floor(
                            (locus_end - locus_start) * ((i + 1) / 30)),
                        locus_end - math.floor(
                            (locus_end - locus_start) * (i / 30)),
                    ]
                    write_bin_to_out(locus, _bin, index)
                    index += 1

                start = locus_start
                for i in range(10):
                    _bin = [start - (i + 1) * 250, start - i * 250]
                    write_bin_to_out(locus, _bin, index)
                    index += 1

        # Make 100-bp bins in a 5 kb window
        elif locus_group_obj.group_type in ['promoter', 'enhancer']:
            region_start = math.floor((locus_start + locus_end) / 2)

            if locus.strand == '+' or locus.strand is None:
                start = region_start - 2500
                for i in range(50):
                    _bin = [start + i * 100, start + (i + 1) * 100]
                    write_bin_to_out(locus, _bin, i)

            elif locus.strand == '-':
                start = region_start + 2500
                for i in range(50):
                    _bin = [start - (i + 1) * 100, start - i * 100]
                    write_bin_to_out(locus, _bin, i)

    OUT.flush()
