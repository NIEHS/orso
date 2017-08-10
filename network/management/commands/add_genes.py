from django.core.management.base import BaseCommand
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
        parser.add_argument('annotation_pk', type=int)
        parser.add_argument('annotation_gtf', type=str)
        parser.add_argument('annotation_table', type=str)

    def handle(self, *args, **options):
        annotation_obj = models.Annotation.objects.get(
            pk=options['annotation_pk'])

        transcripts = dict()
        with open(options['annotation_gtf']) as f:
            for line in f:
                line_split = line.strip().split('\t')

                chromosome = line_split[0]
                _type = line_split[2]
                (start, end) = int(line_split[3]), int(line_split[4])
                strand = line_split[6]
                detail = line_split[8]

                if chromosome in CHROM_LIST and _type == 'exon':
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

        #  Add objects for genes and transcripts
        gene_set = set()
        transcript_to_gene = dict()

        with open(options['annotation_table']) as f:
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
            gene = models.Gene.objects.get(
                name=transcript_to_gene[tr_name],
                annotation=annotation_obj,
            )
            models.Transcript.objects.create(
                name=tr_name,
                gene=gene,
                start=transcript['start'],
                end=transcript['end'],
                chromosome=transcript['chromosome'],
                strand=transcript['strand'],
                exons=transcript['exons'],
            )
