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


def add_gene_loci(assembly_name, annotation_gtf, annotation_table):
    # TODO: Add loci just from the annotation table. The GTF is unnecessary.
    '''
    Add loci from annotated genes to database.
    '''
    # Exception will be raised if assembly does not exist
    assembly_obj = models.Assembly.objects.get(name=assembly_name)

    # Create Annotation
    annotation_obj = models.Annotation.objects.create(
        name=assembly_name + ' RefSeq',
        assembly=assembly_obj,
    )

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

    # Get transcripts from annotation file
    transcripts = dict()
    with open(annotation_gtf) as f:
        for line in f:
            line_split = line.strip().split('\t')

            chromosome = line_split[0]
            start = int(line_split[3])
            end = int(line_split[4])
            strand = line_split[6]
            detail = line_split[8]

            if chromosome in CHROM_LIST:
                transcript_id = detail.split(
                    'transcript_id')[1].split(';')[0].split('"')[1]

                _id = (transcript_id, chromosome)
                if _id not in transcripts:
                    transcripts[_id] = {
                        'chromosome': chromosome,
                        'strand': strand,
                        'exons': [],
                    }
                transcripts[_id]['exons'].append((start, end))

    for transcript in transcripts.values():
        transcript['exons'].sort(key=lambda x: x[0])
        transcript['start'] = transcript['exons'][0][0]
        transcript['end'] = transcript['exons'][-1][1]

    gene_set = set()
    transcript_to_gene = dict()

    # Create gene objects and find associated transcripts
    with open(annotation_table) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for line in reader:
            gene_name = line['name2']
            transcript_name = line['name']

            gene_set.add(gene_name)
            transcript_to_gene[transcript_name] = gene_name

    gene_objs = []
    for gene in gene_set:
        gene_objs.append(models.Gene(
            name=gene,
            annotation=annotation_obj,
        ))
    models.Gene.objects.bulk_create(gene_objs)

    # Create transcripts and associated loci
    for tr_id, transcript in transcripts.items():
        tr_name = tr_id[0].split('_dup')[0]

        associated_gene_name = transcript_to_gene[tr_name]
        associated_gene = models.Gene.objects.get(
            name=associated_gene_name,
            annotation=annotation_obj,
        )

        #  Add promoter locus
        if transcript['strand'] == '+':
            _region = [transcript['start'], transcript['start']]
        elif transcript['strand'] == '-':
            _region = [transcript['end'], transcript['end']]
        else:
            raise ValueError('Transcript is without strand value.')
        promoter_locus = models.Locus.objects.create(
            group=promoter_group,
            strand=transcript['strand'],
            chromosome=transcript['chromosome'],
            regions=[_region],
        )

        # Add genebody locus
        genebody_locus = models.Locus.objects.create(
            group=genebody_group,
            strand=transcript['strand'],
            chromosome=transcript['chromosome'],
            regions=[[transcript['start'], transcript['end']]],
        )

        # Add mRNA locus
        mRNA_locus = models.Locus.objects.create(
            group=mRNA_group,
            strand=transcript['strand'],
            chromosome=transcript['chromosome'],
            regions=transcript['exons'],
        )

        models.Transcript.objects.create(
            name=tr_name,
            gene=associated_gene,
            start=transcript['start'],
            end=transcript['end'],
            chromosome=transcript['chromosome'],
            strand=transcript['strand'],
            exons=transcript['exons'],
            promoter_locus=promoter_locus,
            genebody_locus=genebody_locus,
            mRNA_locus=mRNA_locus,
        )


def add_enhancer_loci(assembly_name, enhancer_bed_file):
    # Create Assembly if it does not exist
    if models.Assembly.objects.filter(name=assembly_name).exists():
        assembly_obj = models.Assembly.objects.get(name=assembly_name)
    else:
        assembly_obj = models.Assembly.objects.create(name=assembly_name)

    # Create Annotation
    annotation_obj = models.Annotation.objects.create(
        name=assembly_name + ' Enhancers',
        assembly=assembly_obj,
    )

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

            locus = models.Locus.objects.create(
                group=enhancer_group,
                chromosome=chromosome,
                regions=[[start, end]],
            )

            models.Enhancer.objects.create(
                annotation=annotation_obj,
                name=name,
                chromosome=chromosome,
                start=start,
                end=end,
                locus=locus,
            )


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('loci_info_file', type=str)
        '''
        Format of each annotation_list row:

        genes:
        genes ASSEMBLY_NAME PATH_TO_GTF PATH_TO_UCSC_ANNOTATION_TABLE

        enhancers:
        enhancers ASSEMBLY_NAME PATH_TO_BED
        '''

    def handle(self, *args, **options):
        with open(options['loci_info_file']) as _list:
            for line in _list:

                _id = line.strip().split()[0]

                if _id == 'genes':

                    assembly, annotation_file, annotation_table = \
                        line.strip().split()[1:]
                    add_gene_loci(assembly, annotation_file, annotation_table)

                elif _id == 'enhancers':

                    assembly, enhancer_bed = line.strip().split()[1:]
                    add_enhancer_loci(assembly, enhancer_bed)
