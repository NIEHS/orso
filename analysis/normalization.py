from collections import defaultdict


def normalize_locus_intersection_values(locus_values, locus_bed_path):
    '''
    Take transcription intersection counts and normalize to pseudoTPM for
    genebody, coding regions, and promoter. Add normalized values to dict.

    Here, normalized values:
    TPM = value / (1K / length) / (1M / total sum)
    '''
    normalized_values = dict()
    locus_cpk = dict()  # cpk = counts per kb
    lengths = get_feature_lengths_from_locus_bed(locus_bed_path)

    for locus_pk, value in locus_values.items():
        length = lengths[locus_pk]
        locus_cpk[locus_pk] = value * (1000.0 / length)

    cpk_sum = sum(locus_cpk.values())

    for locus_pk, value in locus_cpk.items():
        normalized_values[locus_pk] = value / (cpk_sum / 1E6)

    return normalized_values


def get_feature_lengths_from_locus_bed(bed_path):
    '''
    From a locus BED file, get the length of locus features.
    '''
    lengths = defaultdict(int)
    with open(bed_path) as f:
        for line in f:
            chromosome, start, end, name = line.strip().split()[:4]
            pk = name.split('_')[0]
            lengths[pk] += (int(start) - int(end))
    return lengths
