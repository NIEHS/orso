from collections import defaultdict
from tempfile import NamedTemporaryFile

import numpy as np
from celery import group
from celery.decorators import task

from network.tasks.analysis.utils import \
    call_bigwig_average_over_bed, generate_intersection_df
from network import models


def get_locus_values(loci, locus_bed_path, ambiguous_bigwig=None,
                     plus_bigwig=None, minus_bigwig=None):
    '''
    Finds coverage values for each transcript.

    loci - Dict of locus objects from models.LocusGroup.get_loci_dict()
    locus_bed_bed - Path to BED file with loci intervals.
    '''
    if plus_bigwig and minus_bigwig:
        plus_tab = NamedTemporaryFile(mode='w')
        minus_tab = NamedTemporaryFile(mode='w')

        call_bigwig_average_over_bed(
            plus_bigwig,
            locus_bed_path,
            plus_tab.name,
        )
        call_bigwig_average_over_bed(
            minus_bigwig,
            locus_bed_path,
            minus_tab.name,
        )

        plus_tab.flush()
        minus_tab.flush()

        return reconcile_stranded_coverage(
            loci,
            read_bigwig_average_over_bed_tab_file(loci, plus_tab.name),
            read_bigwig_average_over_bed_tab_file(loci, minus_tab.name),
        )

    elif ambiguous_bigwig:
        tab = NamedTemporaryFile(mode='w')

        call_bigwig_average_over_bed(
            ambiguous_bigwig,
            locus_bed_path,
            tab.name,
        )
        tab.flush()

        out_values = read_bigwig_average_over_bed_tab_file(loci, tab.name)
        return out_values

    else:
        raise ValueError('Improper bigWig files specified.')


def read_bigwig_average_over_bed_tab_file(loci, tab_file_path):
    '''
    Read values in bigWigAverageOverBed output file into dict.
    '''
    pk_to_value = defaultdict(float)
    with open(tab_file_path) as f:
        for line in f:
            name, size, covered, value_sum, mean, mean0 = line.strip().split()
            locus_pk = int(name.split('_')[0])

            # Rarely, ENCODE uses nan in their bigWig files; if found, set to 0
            if value_sum == 'nan':
                value_sum = 0

            pk_to_value[locus_pk] += float(value_sum)

    locus_values = dict()
    for locus in loci:
        locus_values[locus] = pk_to_value[locus.pk]

    return locus_values


def reconcile_stranded_coverage(loci, plus_values, minus_values):
    '''
    Considering plus and minus strand coverage values, return only coverage
    values of the appropriate strand.
    '''
    reconciled = dict()
    for locus in loci:
        if locus.strand is None:
            reconciled[locus] = plus_values[locus] + \
                minus_values[locus]
        elif locus.strand == '+':
            reconciled[locus] = plus_values[locus]
        elif locus.strand == '-':
            reconciled[locus] = minus_values[locus]
    return reconciled


def generate_locusgroup_bed(locus_group, output_file_obj):
    '''
    Write a BED file to output_file_obj containing entries for each locus in a
    locus group.
    '''
    OUT = output_file_obj

    def write_to_out(locus, interval, index):
        '''
        Write interval to OUT in BED6 format
        '''
        if locus.strand:
            strand = locus.strand
        else:
            strand = '.'
        OUT.write('\t'.join([
            locus.chromosome,
            str(interval[0] - 1),
            str(interval[1]),
            '{}_{}'.format(str(locus.pk), str(index)),
            '0',
            strand,
        ]) + '\n')

    for locus in models.Locus.objects.filter(group=locus_group):
        for i, region in enumerate(locus.regions):
            write_to_out(locus, region, i)

    OUT.flush()


def set_selected_transcripts_for_genes():
    # Find all annotations with genes
    annotations = models.Annotation.objects.filter(
        gene__isnull=False).distinct()

    # Run in parallel for each annotation found
    job = group(_set_selected_transcripts_for_genes.s(annotation.pk)
                for annotation in annotations)
    job.apply_async()


@task
def _set_selected_transcripts_for_genes(annotation_pk):
    # Given the annotation, find the appropriate locus_group and
    # experiment_type
    annotation = models.Annotation.objects.get(pk=annotation_pk)
    experiment_type = models.ExperimentType.objects.get(name='RNA-seq')

    datasets = models.Dataset.objects.filter(
        experiment__project__name='ENCODE',
        experiment__experiment_type=experiment_type,
        assembly=annotation.assembly,
    )

    if datasets.exists():  # Check to ensure RNA-seq data exists
        locus_group = models.LocusGroup.objects.filter(
            assembly=annotation.assembly, group_type='mRNA')
        df = generate_intersection_df(locus_group, experiment_type,
                                      datasets=datasets)

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
