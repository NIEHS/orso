from network import models


def normalize_transcript_intersection_values(transcript_values,
                                             promoter_size=5000):
    '''
    Take transcription intersection counts and normalize to pseudoTPM for
    genebody, coding regions, and promoter. Add normalized values to dict.

    Here, normalized values:
    TPM = value / (1K / length) / (1M / total sum)
    '''
    normalized_values = dict()

    ts_genebody_cpk = dict()
    ts_coding_cpk = dict()
    ts_promoter_cpk = dict()

    for transcript, value_dict in transcript_values.items():
        length = transcript.end - transcript.start
        ts_genebody_cpk[transcript] = \
            value_dict['genebody'] * (1000.0 / length)

        ts_promoter_cpk[transcript] = \
            value_dict['promoter'] * (1000.0 / promoter_size)

        coding_sum = 0
        coding_length = 0
        for i, exon in enumerate(transcript.exons):
            coding_length += (exon[1] - exon[0])
            coding_sum += value_dict['exons'][i]
        ts_coding_cpk[transcript] = \
            coding_sum * (1000.0 / coding_length)

    genebody_cpk_sum = sum(ts_genebody_cpk.values())
    promoter_cpk_sum = sum(ts_promoter_cpk.values())
    coding_cpk_sum = sum(ts_coding_cpk.values())

    for transcript in transcript_values.keys():

        normalized_values[transcript] = dict()
        norm = normalized_values[transcript]

        norm['genebody'] = ts_genebody_cpk[transcript] \
            / (genebody_cpk_sum / 1E6)
        norm['promoter'] = ts_promoter_cpk[transcript] \
            / (promoter_cpk_sum / 1E6)
        norm['coding'] = ts_coding_cpk[transcript] \
            / (coding_cpk_sum / 1E6)

    return normalized_values


def normalize_dataset(dataset):
    '''
    Apply normalize_transcript_intersection_values to TranscriptIntersection
    objects associated with a Dataset object.
    '''
    transcript_values = dict()
    for intersection in models.TranscriptIntersection.objects.filter(
            dataset=dataset,
            transcript__gene__annotation=dataset.assembly.geneannotation):
        transcript_values[intersection.transcript] = {
            'genebody': intersection.genebody_value,
            'promoter': intersection.promoter_value,
            'exons': intersection.exon_values,
        }
    return normalize_transcript_intersection_values(transcript_values)
