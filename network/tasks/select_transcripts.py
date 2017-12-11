from celery import group
from celery.decorators import task

from network import models


@task
def set_selected_transcripts_for_genes():
    '''
    For each gene set the selected transcript using expression values.
    '''
    job = group(_set_selected_transcript_for_gene.s(g.pk)
                for g in models.Gene.objects.all())
    job.apply_async()


@task
def _set_selected_transcript_for_gene(gene_pk):
    '''
    Set selected transcript for a single gene.
    '''
    gene = models.Gene.objects.get(pk=gene_pk)
    transcripts = models.Transcript.objects.filter(gene=gene).order_by(
        'name', 'pk')

    if transcripts:
        # If no DatasetIntersection object exists for transcripts, the
        # following will return None
        transcript_w_highest_expression = \
            gene.get_transcript_with_highest_expression()

        if transcript_w_highest_expression:
            transcript = transcript_w_highest_expression
        else:
            transcript = transcripts[0]
    else:
        transcript = None

    gene.selected_transcript = transcript
    gene.save()
