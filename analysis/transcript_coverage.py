from tempfile import NamedTemporaryFile

from analysis.utils import call_bigwig_average_over_bed
from network import models


def get_transcript_values(transcripts, transcript_bed, ambiguous_bigwig=None,
                          plus_bigwig=None, minus_bigwig=None):
    '''
    Finds coverage values for each transcript.

    transcripts - List of transcript objects.
    transcript_bed - Path to BED file with transcript intervals.
    '''
    if plus_bigwig and minus_bigwig:
        plus_tab = NamedTemporaryFile(mode='w')
        minus_tab = NamedTemporaryFile(mode='w')

        call_bigwig_average_over_bed(
            plus_bigwig,
            transcript_bed,
            plus_tab.name,
        )
        call_bigwig_average_over_bed(
            minus_bigwig,
            transcript_bed,
            minus_tab.name,
        )

        plus_tab.flush()
        minus_tab.flush()

        return reconcile_stranded_coverage(
            transcripts,
            read_bigwig_average_over_bed_tab_file(plus_tab.name),
            read_bigwig_average_over_bed_tab_file(minus_tab.name),
        )

    elif ambiguous_bigwig:
        tab = NamedTemporaryFile(mode='w')

        call_bigwig_average_over_bed(
            ambiguous_bigwig,
            transcript_bed,
            tab.name,
        )
        tab.flush()

        out_values = read_bigwig_average_over_bed_tab_file(tab.name)
        return out_values

    else:
        raise ValueError('Improper bigWig files specified.')


def exons_to_introns(exon_list):
    '''
    Take a list of exons ([start, end]) and return a list of introns
    ([start, end]).
    '''
    introns = []
    for i in range(len(exon_list) - 1):
        introns.append([exon_list[i][1] + 1, exon_list[i + 1][0] - 1])
    return introns


def generate_transcript_bed(transcripts, chrom_sizes_dict, output_file_obj):
    '''
    Write a BED file to output_file_obj containing entries for each transcript
    model in transcripts.
    '''
    OUT = output_file_obj

    def write_to_out(transcript, interval, name):
        '''
        Write interval to OUT in BED6 format
        '''
        OUT.write('\t'.join([
            transcript.chromosome,
            str(interval[0] - 1),
            str(interval[1]),
            '{}_{}'.format(str(transcript.pk), name),
            '0',
            transcript.strand,
        ]) + '\n')

    for transcript in transcripts:

        chrom = transcript.chromosome

        genebody = [transcript.start, transcript.end]
        introns = exons_to_introns(transcript.exons)
        if transcript.strand == '+':
            promoter = [
                max(transcript.start - 2500, 1),
                min(transcript.start + 2499, chrom_sizes_dict[chrom]),
            ]
        elif transcript.strand == '-':
            promoter = [
                max(transcript.end - 2499, 1),
                min(transcript.end + 2500, chrom_sizes_dict[chrom]),
            ]

        write_to_out(transcript, genebody, 'genebody')
        write_to_out(transcript, promoter, 'promoter')
        for i, exon in enumerate(transcript.exons):
            write_to_out(transcript, exon, 'exon_{}'.format(str(i)))
        for i, intron in enumerate(introns):
            write_to_out(transcript, intron, 'intron_{}'.format(str(i)))

    OUT.flush()


def read_bigwig_average_over_bed_tab_file(tab_file_name):
    '''
    Read values in bigWigAverageOverBed output file into dict.
    '''
    def place_value(_dict, feature_name, value):
        '''
        Based on feature_name, place value in the appropriate location in
        _dict.
        '''
        if 'exon' in feature_name or 'intron' in feature_name:
            key = '{}s'.format(feature_name.split('_')[0])
            index = int(feature_name.split('_')[1])
            for _ in range((index + 1) - len(_dict[key])):
                _dict[key].append(0.0)
            _dict[key][index] = value
        else:
            _dict[feature_name] = value

    transcript_values = dict()
    with open(tab_file_name) as f:
        for line in f:
            name, size, covered, value_sum, mean, mean0 = line.strip().split()
            pk = int(name.split('_')[0])
            feature_name = name.split('{}_'.format(str(pk)))[1]
            if pk not in transcript_values:
                transcript_values[pk] = {
                    'genebody': None,
                    'promoter': None,
                    'exons': [],
                    'introns': [],
                }
            place_value(
                transcript_values[pk], feature_name, float(value_sum))

    pk_to_transcript = \
        models.Transcript.objects.in_bulk(list(transcript_values.keys()))

    out_dict = dict()
    for pk, values in transcript_values.items():
        out_dict[pk_to_transcript[pk]] = values

    return out_dict


def reconcile_stranded_coverage(transcripts, plus_values, minus_values):
    '''
    Considering plus and minus strand coverage values, return only coverage
    values of the appropriate strand.
    '''
    transcript_values = {}

    for transcript in transcripts:
        if transcript.strand == '+':
            transcript_values[transcript] = plus_values[transcript]
        elif transcript.strand == '-':
            transcript_values[transcript] = minus_values[transcript]

    return transcript_values
