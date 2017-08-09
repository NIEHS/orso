import numpy

from network import models


def select_transcript_by_expression(gene):
    '''
    For the gene object, consider all transcripts and select based on highest
    median expression for genebody, promoter, or coding regions.
    '''
    transcripts = models.Transcript.objects.filter(gene=gene)
    transcript_median_expression = dict()
    for transcript in transcripts:
        genebody_values = []
        promoter_values = []
        coding_values = []

        intersections = \
            models.TranscriptIntersection.objects.filter(transcript=transcript)
        for intersection in intersections:
            genebody_values.append(intersection.normalized_genebody_value)
            promoter_values.append(intersection.normalized_promoter_value)
            coding_values.append(intersection.normalized_coding_value)
        transcript_median_expression[transcript] = {
            'genebody': numpy.median(genebody_values),
            'promoter': numpy.median(promoter_values),
            'coding': numpy.median(coding_values),
        }

    if transcript_median_expression:
        gene.highest_exp_genebody_transcript = \
            sorted(transcript_median_expression.items(),
                   key=lambda x: (-x[1]['genebody'], x[0].pk))[0][0]
        gene.highest_exp_promoter_transcript = \
            sorted(transcript_median_expression.items(),
                   key=lambda x: (-x[1]['promoter'], x[0].pk))[0][0]
        gene.highest_exp_coding_transcript = \
            sorted(transcript_median_expression.items(),
                   key=lambda x: (-x[1]['coding'], x[0].pk))[0][0]
    else:
        gene.highest_exp_genebody_transcript = None
        gene.highest_exp_promoter_transcript = None
        gene.highest_exp_coding_transcript = None

    gene.save()
