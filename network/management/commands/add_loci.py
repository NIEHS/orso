import csv

from django.core.management.base import BaseCommand
from network import models

CHROM_LIST = []
#  Addresses human and mouse
for i in range(1, 23):
    CHROM_LIST.append('chr{}'.format(str(i)))
#  Addresses Drosophila m.
for i in [
    '2L', '2LHet', '2R', '2RHet', '3L', '3LHet', '3R', '3RHet', '4', 'X',
    'XHet', 'YHet', 'U', 'Uextra',
]:
    CHROM_LIST.append('chr{}'.format(i))
#  Addresses C. elegans
for i in [
    'I', 'II', 'III', 'IV', 'V', 'X',
]:
    CHROM_LIST.append('chr{}'.format(i))


def add_gene_loci(assembly_name, annotation_name, annotation_table):
    '''
    Add loci from annotated genes to database.
    '''
    # Exception will be raised if assembly does not exist
    assembly_obj = models.Assembly.objects.get(name=assembly_name)

    # Create Annotation
    annotation_obj = models.Annotation.objects.get_or_create(
        name=annotation_name,
        assembly=assembly_obj,
    )[0]

    # Create LocusGroup objects
    promoter_group = models.LocusGroup.objects.create(
        assembly=assembly_obj,
        group_type='promoter',
    )
    genebody_group = models.LocusGroup.objects.create(
        assembly=assembly_obj,
        group_type='genebody',
    )
    mRNA_group = models.LocusGroup.objects.create(
        assembly=assembly_obj,
        group_type='mRNA',
    )

    with open(annotation_table) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:

            transcript_name = row['name']
            gene_name = row['name2']
            chromosome = row['chrom']
            strand = row['strand']
            transcript_start = int(row['txStart'])
            transcript_end = int(row['txEnd'])

            exons = []
            for start, end in zip(
                row['exonStarts'].split(',')[:-1],
                row['exonEnds'].split(',')[:-1],
            ):
                exons.append([int(start), int(end)])

            if chromosome in CHROM_LIST:

                # Get or create gene
                gene = models.Gene.objects.get_or_create(
                    name=gene_name,
                    annotation=annotation_obj,
                )[0]

                # Create transcript
                transcript = models.Transcript.objects.create(
                    gene=gene,
                    name=transcript_name,
                    chromosome=chromosome,
                    strand=strand,
                    start=transcript_start,
                    end=transcript_end,
                    exons=exons,
                )

                # Create promoter locus
                if strand == '+':
                    regions = [[transcript_start, transcript_end]]
                elif strand == '-':
                    regions = [[transcript_end, transcript_end]]
                else:
                    raise ValueError('Transcript is without strand value.')
                models.Locus.objects.create(
                    group=promoter_group,
                    transcript=transcript,
                    strand=strand,
                    chromosome=chromosome,
                    regions=regions,
                )

                # Create genebody locus
                models.Locus.objects.create(
                    group=genebody_group,
                    transcript=transcript,
                    strand=strand,
                    chromosome=chromosome,
                    regions=[[transcript_start, transcript_end]],
                )

                # Create mRNA locus
                models.Locus.objects.create(
                    group=mRNA_group,
                    transcript=transcript,
                    strand=strand,
                    chromosome=chromosome,
                    regions=exons,
                )


def add_enhancer_loci(assembly_name, annotation_name, enhancer_bed_file):
    '''
    Add enhancer loci from BED file.
    '''
    # Exception will be raised if assembly does not exist
    assembly_obj = models.Assembly.objects.get(name=assembly_name)

    # Create Annotation
    annotation_obj = models.Annotation.objects.get_or_create(
        name=annotation_name,
        assembly=assembly_obj,
    )[0]

    # Create LocusGroup objects
    enhancer_group = models.LocusGroup.objects.create(
        assembly=assembly_obj,
        group_type='enhancer',
    )

    # Iterate through BED file and create enhancers with loci
    with open(enhancer_bed_file) as f:
        for line in f:
            chromosome, start, end, name = line.strip().split()
            start = int(start) + 1
            end = int(end)

            enhancer = models.Enhancer.objects.create(
                annotation=annotation_obj,
                name=name,
                chromosome=chromosome,
                start=start,
                end=end,
            )

            models.Locus.objects.create(
                group=enhancer_group,
                enhancer=enhancer,
                chromosome=chromosome,
                regions=[[start, end]],
            )


class Command(BaseCommand):
    help = '''
        Adds loci and associated genes/enhancers to the database.

        If an assembly object does not exist for the assembly name provided, an
        exception will be raised.

        For genes, the loci file is expected to be a table from UCSC table
        browser with transcript and exon information. The following fields are
        required:

        transcript name ("name"),
        gene name ("name2"),
        chromosome ("chrom"),
        strand ("strand"),
        transcript start ("txStart"),
        transcript end ("txEnd"),
        exon starts ("exonStarts"), and
        exon ends ("exonEnds").

        For enhancers, the loci file is expected to be a BED4 file.
    '''

    def add_arguments(self, parser):
        parser.add_argument('loci_type', type=str,
                            choices=['genes', 'enhancers'])
        parser.add_argument('assembly_obj_name', type=str)
        parser.add_argument('annotation_name', type=str)
        parser.add_argument('loci_file', type=str)

    def handle(self, *args, **options):
        if options['loci_type'] == 'genes':
            add_gene_loci(
                options['assembly_obj_name'],
                options['annotation_name'],
                options['loci_file'],
            )
        elif options['loci_type'] == 'enhancers':
            add_enhancer_loci(
                options['assembly_obj_name'],
                options['annotation_name'],
                options['loci_file'],
            )
