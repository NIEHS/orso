import numpy

from network import models


def coeff_variance(array_like):
    '''
    Return coefficient of variance.
    '''
    if numpy.isclose(numpy.mean(array_like), 0):
        return 0.0
    else:
        return numpy.std(array_like) / abs(numpy.mean(array_like))


def rank_genes_by_variance(annotation):
    '''
    For the annotation, rank genes by variance considering transcripts with the
    highest median expression. Consider genebody, promoter, and coding regions.
    '''
    genes = models.Gene.objects.filter(annotation=annotation)[:1000]

    genebody_values = dict()
    promoter_values = dict()
    coding_values = dict()

    for gene in genes:
        if gene.highest_exp_genebody_transcript:
            _tr = gene.highest_exp_genebody_transcript
            genebody_values[gene] = _tr.get_intersection_variance()['genebody']

        if gene.highest_exp_promoter_transcript:
            _tr = gene.highest_exp_promoter_transcript
            promoter_values[gene] = _tr.get_intersection_variance()['promoter']

        if gene.highest_exp_coding_transcript:
            _tr = gene.highest_exp_coding_transcript
            coding_values[gene] = _tr.get_intersection_variance()['coding']

    for i, (gene, value) in enumerate(sorted(
            genebody_values.items(), key=lambda x: (-x[1], x[0].pk))):
        gene.genebody_var_rank = i
        gene.save()

    for i, (gene, value) in enumerate(sorted(
            promoter_values.items(), key=lambda x: (-x[1], x[0].pk))):
        gene.promoter_var_rank = i
        gene.save()

    for i, (gene, value) in enumerate(sorted(
            coding_values.items(), key=lambda x: (-x[1], x[0].pk))):
        gene.coding_var_rank = i
        gene.save()
