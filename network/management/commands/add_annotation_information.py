from django.core.management.base import BaseCommand
from collections import defaultdict
from network import models

import csv

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


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('annotation_list', type=str)
        '''
        Format of each annotation_list row:
        assembly path_to_annotation path_annotation_table
        '''

    def handle(self, *args, **options):
        with open(options['annotation_list']) as _list:
            for line in _list:
                assembly, annotation_file, annotation_table = \
                    line.strip().split()

                #  Create Assembly
                assembly_obj = models.Assembly.objects.create(
                    name=assembly,
                )

                #  Create Annotation
                annotation_obj = models.Annotation.objects.create(
                    name=assembly + '_RefSeq',
                    gtf_file=annotation_file,
                    assembly=assembly_obj,
                )
                assembly_obj.annotation = annotation_obj
                assembly_obj.save()

                #  Get transcripts from annotation file
                transcripts = dict()
                with open(annotation_file) as f:
                    for line in f:
                        line_split = line.strip().split('\t')

                        chromosome = line_split[0]
                        (start, end) = int(line_split[3]), int(line_split[4])
                        strand = line_split[6]

                        detail = line_split[8]

                        if chromosome in CHROM_LIST:
                            transcript_id = detail.split(
                                'transcript_id')[1].split(';')[0].split('"')[1]

                            if transcript_id not in transcripts:
                                transcripts[(transcript_id, chromosome)] = {
                                    'chromosome': chromosome,
                                    'strand': strand,
                                    'exons': [],
                                }

                            _id = (transcript_id, chromosome)
                            transcripts[_id]['exons'].append((start, end))

                for transcript in transcripts.values():
                    transcript['exons'].sort(key=lambda x: x[0])
                    transcript['start'] = transcript['exons'][0][0]
                    transcript['end'] = transcript['exons'][-1][1]

                #  Create promoter BED from transcripts
                tss_dict = defaultdict(set)
                output_file = 'data/genomic_regions/' + \
                    assembly + '_RefSeq_promoters.bed'
                for tr_id, transcript in transcripts.items():
                    chromosome = transcript['chromosome']
                    strand = transcript['strand']
                    if strand == '+':
                        start = transcript['start']
                    if strand == '-':
                        start = transcript['end']
                    name = tr_id[0]

                    tss_dict[(chromosome, strand, start)].add(name)

                entry_num = 0
                with open(output_file, 'w') as OUTPUT:
                    for tr_id, names in sorted(
                        tss_dict.items(),
                        key=lambda x: (
                            CHROM_LIST.index(x[0][0]),
                            x[0][2]
                        )
                    ):
                        tr_names = ','.join(
                            sorted(list(names), key=lambda x: int(x.split('_')[1])))  # noqa
                        chromosome = tr_id[0]
                        entry_name = '{}|{}|{}'.format(
                            str(entry_num), tr_names, chromosome)

                        start = tr_id[2]
                        strand = tr_id[1]

                        OUTPUT.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                            chromosome,
                            str(start - 1),
                            str(start),
                            entry_name,
                            '0',
                            strand,
                        ))

                        entry_num += 1

                #  Add GenomicRegions object for promoters
                models.GenomicRegions.objects.create(
                    name=assembly + '_RefSeq_promoters',
                    assembly=assembly_obj,
                    bed_file=output_file,
                    short_label='Promoters',
                )

                #  Add objects for genes and transcripts
                gene_set = set()
                transcript_to_gene = dict()

                with open(annotation_table) as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for line in reader:
                        gene_name = line['name2']
                        transcript_name = line['name']

                        gene_set.add(gene_name)
                        transcript_to_gene[transcript_name] = gene_name

                for gene in gene_set:
                    models.Gene.objects.create(
                        name=gene,
                        annotation=annotation_obj,
                    )

                for tr_id, transcript in transcripts.items():
                    tr_name = tr_id[0].split('_dup')[0]
                    associated_gene_name = transcript_to_gene[tr_name]
                    associated_gene = \
                        models.Gene.objects.get(
                            name=associated_gene_name,
                            annotation=annotation_obj,
                        )

                    models.Transcript.objects.create(
                        name=tr_name,
                        gene=associated_gene,
                        start=transcript['start'],
                        end=transcript['end'],
                        chromosome=transcript['chromosome'],
                        strand=transcript['strand'],
                        exons=transcript['exons'],
                    )
