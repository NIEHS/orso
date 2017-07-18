from subprocess import call
from tempfile import NamedTemporaryFile

from analysis.utils import call_bigwig_average_over_bed


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

        return read_bigwig_average_over_bed_tab_file(tab.name)

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
    transcript_values = dict()

    with open(tab_file_name) as f:
        for line in f:

            name, size, covered, value_sum, mean, mean0 = line.strip().split()
            transcript_pk = int(name.split('_')[0])
            feature_name = name.split('{}_'.format(str(transcript_pk)))[1]

            if transcript_pk not in transcript_values:
                transcript_values[transcript_pk] = {}
            transcript_values[transcript_pk][feature_name] = float(value_sum)

    for pk, value_dict in transcript_values.items():

        exon_count = 0
        intron_count = 0
        for feature_name in value_dict.keys():
            if 'exon' in feature_name:
                exon_count += 1
            elif 'intron' in feature_name:
                intron_count += 1

        exons = [0] * exon_count
        introns = [0] * intron_count

        keys_to_remove = []
        for feature_name, value in value_dict.items():
            if 'exon' in feature_name:
                exon_index = int(feature_name.split('exon_')[1])
                exons[exon_index] = value
                keys_to_remove.append(feature_name)
            elif 'intron' in feature_name:
                intron_index = int(feature_name.split('intron_')[1])
                introns[intron_index] = value
                keys_to_remove.append(feature_name)

        transcript_values[pk]['exons'] = exons
        transcript_values[pk]['introns'] = introns

        for key in keys_to_remove:
            del transcript_values[pk][key]

    return transcript_values


def reconcile_stranded_coverage(transcripts, plus_values, minus_values):
    '''
    Considering plus and minus strand coverage values, return only coverage
    values of the appropriate strand.
    '''
    transcript_values = {}

    for transcript in transcripts:
        if transcript.strand == '+':
            transcript_values[transcript.pk] = plus_values[transcript.pk]
        elif transcript.strand == '-':
            transcript_values[transcript.pk] = minus_values[transcript.pk]

    return transcript_values
