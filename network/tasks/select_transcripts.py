import numpy as np
from celery import group
from celery.decorators import task

from network.tasks.analysis.utils import generate_intersection_df
from network import models


@task
def set_selected_transcripts_for_genes_json():
    # Find all annotations with genes
    annotations = models.Annotation.objects.filter(
        gene__isnull=False).distinct()

    # Run in parallel for each annotation found
    job = group(_set_selected_transcript_for_genes_json.s(annotation.pk)
                for annotation in annotations)
    job.apply_async()


@task
def _set_selected_transcript_for_genes_json(annotation_pk):
    # Given the annotation, find the appropriate locus_group and
    # experiment_type
    annotation = models.Annotation.objects.get(pk=annotation_pk)
    locus_group = models.LocusGroup.objects.filter(
        assembly=annotation.assembly, group_type='mRNA')
    experiment_type = models.ExperimentType.objects.filter(
        name='RNA-seq')

    if models.DatasetIntersectionJson.objects.filter(
        dataset__experiment__experiment_type=experiment_type,
        locus_group=locus_group,
    ).exists():  # Check to ensure RNA-seq data exists
        df = generate_intersection_df(locus_group, experiment_type)

        for gene in models.Gene.objects.filter(annotation=annotation):
            # Check to ensure transcripts exist for the gene
            if models.Transcript.objects.filter(gene=gene).exists():

                loci = models.Locus.objects.filter(
                    transcript__gene=gene, group=locus_group)

                expression = dict()
                for locus in loci:
                    expression[locus] = np.median(df.loc[locus.pk])

                selected_locus = sorted(
                    expression.items(), key=lambda x: -x[1])[0][0]
                selected_transcript = models.Transcript.objects.get(
                    gene=gene, locus=selected_locus)

                gene.selected_transcript = selected_transcript
                gene.save()


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
