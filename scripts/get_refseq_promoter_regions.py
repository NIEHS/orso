import mysql.connector
import csv
import click

from datetime import datetime
from collections import defaultdict

CHROM_SET = {'chrX'}
#  Addresses human and mouse
for i in range(1, 23):
    CHROM_SET.add('chr{}'.format(str(i)))
#  Addresses Drosophila m.
for i in [
    '2L', '2LHet', '2R', '2RHet', '3L', '3LHet', '3R', '3RHet', '4', 'X',
    'XHet', 'YHet', 'U', 'Uextra',
]:
    CHROM_SET.add('chr{}'.format(i))
#  Addresses C. elegans
for i in [
    'I', 'II', 'III', 'IV', 'V', 'X',
]:
    CHROM_SET.add('chr{}'.format(i))


def getPosition(transcript, coordinate):
    return (
        transcript['chrom'],
        transcript['strand'],
        str(coordinate),
    )


def write_promoters_to_bed(assembly, out_bed):
    cnx = mysql.connector.connect(
        user='genome',
        host='genome-mysql.cse.ucsc.edu',
        database=assembly,
    )

    cursor = cnx.cursor(dictionary=True)
    cursor.execute(
        'SELECT name,name2,exonStarts,exonEnds,strand,chrom FROM refGene')

    transcripts = []
    tss_dict = defaultdict(list)

    for entry in cursor:
        transcripts.append(entry)

    #  Add a count to avoid duplicate entry names
    name_count = defaultdict(int)
    for transcript in transcripts:
        name_count[transcript['name']] += 1
        transcript['name'] = transcript['name'] + ':' + \
            str(name_count[transcript['name']])

    for transcript in transcripts:
        exon_starts = transcript['exonStarts'].decode('utf-8').split(',')[:-1]
        exon_ends = transcript['exonEnds'].decode('utf-8').split(',')[:-1]

        exons = []
        for start, end in zip(exon_starts, exon_ends):
            exons.append([start, end])
        exons.sort(key=lambda x: int(x[0]))

        if transcript['strand'] == '+':
            tss = exons[0][0]
        elif transcript['strand'] == '-':
            tss = exons[-1][1]

        tss_dict[getPosition(transcript, tss)].append(transcript['name'])

    with open(out_bed, 'w') as OUT:
        OUT.write('track name={} description="{}"\n'.format(
            '{}_RefSeq_TSSs'.format(assembly),
            'Accessed on {}'.format(datetime.now().date()),
        ))

        writer = csv.writer(OUT, delimiter='\t')
        for pos in sorted(
                tss_dict, key=lambda x: (x[1], x[0], int(x[2]))):
            chrom, strand, coordinate = pos
            if chrom in CHROM_SET:
                joined_name = ','.join(tss_dict[pos])
                writer.writerow([
                    chrom,
                    str(int(coordinate) - 1),
                    str(coordinate),
                    joined_name,
                    '0',
                    strand,
                ])


@click.command()
@click.argument('assembly')
@click.argument('out_bed')
def cli(assembly, out_bed):
    '''
    Generate a BED file of promoter regions from UCSC RefSeq TSSs
    '''
    write_promoters_to_bed(assembly, out_bed)

if __name__ == '__main__':
    cli()
