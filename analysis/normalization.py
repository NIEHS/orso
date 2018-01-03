def normalize_locus_intersection_values(loci, locus_values):
    '''
    Take transcription intersection counts and normalize to pseudoTPM for
    genebody, coding regions, and promoter. Add normalized values to dict.

    Here, normalized values:
    TPM = value / (1K / length) / (1M / total sum)
    '''
    normalized_values = dict()
    locus_cpk = dict()  # cpk = counts per kb

    # Set negative values to zero
    for locus in loci:
        locus_values[locus] = max(locus_values[locus], 0)

    for locus in loci:
        length = 0
        for region in locus.regions:
            length += region[1] - region[0] + 1
        locus_cpk[locus] = locus_values[locus] * (1000.0 / length)

    cpk_sum = sum(locus_cpk.values())

    for locus, value in locus_cpk.items():
        normalized_values[locus] = value / (cpk_sum / 1E6)

    return normalized_values
