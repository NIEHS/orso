import csv
import json
import math
import os

import mysql.connector
from django.conf import settings
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


ASSEMBLIES = [
    ('Homo sapiens', 'GRCh38', 'hg38', 'GRCh38.chrom.sizes'),
    ('Homo sapiens', 'hg19', 'hg19', 'hg19.chrom.sizes'),
    ('Mus musculus', 'mm10', 'mm10', 'mm10.chrom.sizes'),
    ('Mus musculus', 'mm9', 'mm9', 'mm9.chrom.sizes'),
    ('Drosophila melanogaster', 'dm6', 'dm6', 'dm6.chrom.sizes'),
    ('Drosophila melanogaster', 'dm3', 'dm3', 'dm3.chrom.sizes'),
    ('Caenorhabditis elegans', 'ce11', 'ce11', 'ce11.chrom.sizes'),
    ('Caenorhabditis elegans', 'ce10', 'ce10', 'ce11.chrom.sizes'),
]
ENHANCERS = [
    ('GRCh38', 'VISTA', 'GRCh38_vista_enhancers.bed'),
    ('hg19', 'VISTA', 'hg19_vista_enhancers.bed'),
    ('mm10', 'VISTA', 'mm10_vista_enhancers.bed'),
    ('mm9', 'VISTA', 'mm9_vista_enhancers.bed'),
    ('dm6', 'Stark', 'dm6_stark_enhancers.bed'),
    ('dm3', 'Stark', 'dm3_stark_enhancers.bed'),
    ('ce11', 'Chen', 'ce11_chen_enhancers.bed'),
    ('ce10', 'Chen', 'ce10_chen_enhancers.bed'),
]
ONTOLOGIES = [
    ('brenda_tissue_ontology', 'BTO',
        'brenda_tissue_2017-10-17.obo',
        'brenda_tissue_2017-10-17.ac'),
    ('gene_ontology', 'GO',
        'gene_ontology_2017-09-19.obo',
        'gene_ontology_multiorganism.ac'),
    ('epigenetic_modification_ontology', 'EMO',
        'epigenetic_modification_ontology.obo',
        'epigenetic_modification_ontology.ac'),
]


def add_organisms_and_assemblies():

    for organism_name, assembly_name, ucsc_name, chrom_sizes_file_name in \
            ASSEMBLIES:

        organism = models.Organism.objects.get_or_create(
            name=organism_name)[0]

        assembly = models.Assembly.objects.get_or_create(
            name=assembly_name, organism=organism)[0]
        chrom_sizes_path = os.path.join(
            settings.CHROM_SIZES_DIR, chrom_sizes_file_name)
        assembly.read_in_chrom_sizes(chrom_sizes_path)


def add_refseq():

    for organism_name, assembly_name, ucsc_name, chrom_sizes_file_name in \
            ASSEMBLIES:

        assembly = models.Assembly.objects.get(name=assembly_name)
        chrom_sizes = json.loads(assembly.chromosome_sizes)

        annotation = models.Annotation.objects.get_or_create(
            assembly=assembly,
            name='RefSeq'
        )[0]

        promoter_lg = models.LocusGroup.objects.get_or_create(
            assembly=assembly,
            group_type='promoter',
        )[0]
        genebody_lg = models.LocusGroup.objects.get_or_create(
            assembly=assembly,
            group_type='genebody',
        )[0]
        mrna_lg = models.LocusGroup.objects.get_or_create(
            assembly=assembly,
            group_type='mRNA',
        )[0]

        cnx = mysql.connector.connect(
            user='genome',
            host='genome-mysql.cse.ucsc.edu',
            database=ucsc_name,
        )

        cursor = cnx.cursor(dictionary=True)
        cursor.execute(
            'SELECT name,name2,exonStarts,exonEnds,strand,chrom '
            'FROM refGene')

        db_entries = []
        for entry in cursor:
            db_entries.append(entry)

        cnx.close()

        for entry in db_entries:
            if entry['chrom'] in CHROM_LIST:

                gene = models.Gene.objects.get_or_create(
                    annotation=annotation,
                    name=entry['name2']
                )[0]

                exons = list(zip(
                    [int(x) for x in entry['exonStarts']
                        .decode('utf-8').split(',')[:-1]],
                    [int(x) for x in entry['exonEnds']
                        .decode('utf-8').split(',')[:-1]],
                ))

                transcript = models.Transcript.objects.get_or_create(
                    gene=gene,
                    name=entry['name'],
                    chromosome=entry['chrom'],
                    strand=entry['strand'],
                    start=exons[0][0],
                    end=exons[-1][1],
                    exons=exons,
                )[0]

                if entry['strand'] == '+':
                    promoter_regions = [[
                        max(exons[0][0] - 2500, 1),
                        min(
                            exons[0][0] + 2499,
                            chrom_sizes[entry['chrom']],
                        ),
                    ]]
                elif entry['strand'] == '-':
                    promoter_regions = [[
                        max(exons[-1][1] - 2499, 1),
                        min(
                            exons[-1][1] + 2500,
                            chrom_sizes[entry['chrom']],
                        ),
                    ]]

                models.Locus.objects.get_or_create(
                    group=promoter_lg,
                    transcript=transcript,
                    strand=entry['strand'],
                    chromosome=entry['chrom'],
                    regions=promoter_regions,
                )
                models.Locus.objects.get_or_create(
                    group=genebody_lg,
                    transcript=transcript,
                    strand=entry['strand'],
                    chromosome=entry['chrom'],
                    regions=[[exons[0][0], exons[-1][1]]],
                )
                models.Locus.objects.get_or_create(
                    group=mrna_lg,
                    transcript=transcript,
                    strand=entry['strand'],
                    chromosome=entry['chrom'],
                    regions=exons,
                )

        # Select transcripts by name/pk order
        for gene in models.Gene.objects.filter(annotation=annotation):
            if not gene.selected_transcript:
                gene.selected_transcript = models.Transcript.objects.filter(
                    gene=gene).order_by('name', 'pk')[0]
                gene.save()


def add_enhancers():

    for assembly_name, annotation_name, file_name in ENHANCERS:

        assembly = models.Assembly.objects.get(name=assembly_name)
        chrom_sizes = json.loads(assembly.chromosome_sizes)

        annotation = models.Annotation.objects.get_or_create(
            assembly=assembly, name=annotation_name)[0]
        locus_group = models.LocusGroup.objects.get_or_create(
            assembly=assembly, group_type='enhancer')[0]

        file_path = os.path.join(settings.ENHANCERS_DIR, file_name)

        with open(file_path) as f:

            reader = csv.DictReader(f, delimiter='\t', fieldnames=[
                'chrom', 'start', 'end', 'name'])

            for row in reader:
                if row['chrom'] in CHROM_LIST:

                    start = int(row['start']) + 1
                    end = int(row['end'])
                    center = math.floor((start + end) / 2)

                    enhancer = models.Enhancer.objects.get_or_create(
                        annotation=annotation,
                        name=row['name'],
                        chromosome=row['chrom'],
                        start=start,
                        end=end,
                    )[0]

                    models.Locus.objects.get_or_create(
                        enhancer=enhancer,
                        group=locus_group,
                        strand=None,
                        chromosome=row['chrom'],
                        regions=[[
                            max((center - 2500), 1),
                            min(
                                (center + 2499),
                                chrom_sizes[row['chrom']],
                            ),
                        ]]
                    )


def add_ontologies():

    for ontology_name, ontology_type, obo_file_name, ac_file_name \
            in ONTOLOGIES:

        obo_file_path = os.path.join(settings.ONTOLOGY_DIR, obo_file_name)
        ac_file_path = os.path.join(settings.ONTOLOGY_DIR, ac_file_name)

        models.Ontology.objects.get_or_create(
            name=ontology_name,
            ontology_type=ontology_type,
            obo_file=obo_file_path,
            ac_file=ac_file_path,
        )


def create_bed_files():

    assembly_names = [x[1] for x in ASSEMBLIES]

    for lg in models.LocusGroup.objects.filter(
            assembly__name__in=assembly_names):

        lg.create_and_set_metaplot_bed()
        lg.create_and_set_intersection_bed()


class Command(BaseCommand):

    def handle(self, *args, **options):

        # print('Adding organism and assembly objects...')
        # add_organisms_and_assemblies()

        # print('Adding genes and transcripts from RefSeq...')
        # add_refseq()

        # print('Adding enhancers...')
        # add_enhancers()

        # print('Adding ontologies...')
        # add_ontologies()

        print('Adding BED files...')
        create_bed_files()
