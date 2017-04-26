import click

#  TODO: add to utils
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


def write_table_to_bed(table, bed):
    with open(table) as f, open(bed, 'w') as OUT:
        for line in f:
            if line[0] == '#':  # Check for header
                pass
            else:
                line_split = line.strip().split()

                chromosome = line_split[2]
                strand = line_split[3]
                gene_start = line_split[4]
                gene_end = line_split[5]
                common_name = line_split[12]

                if chromosome in CHROM_SET:
                    OUT.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        chromosome,
                        gene_start,
                        gene_end,
                        common_name,
                        '0',
                        strand,
                    ))


@click.command()
@click.argument('annotation_table')
@click.argument('out_bed')
def cli(annotation_table, out_bed):
    '''
    Generate a BED file of promoter regions from UCSC RefSeq TSSs
    '''
    write_table_to_bed(annotation_table, out_bed)

if __name__ == '__main__':
    cli()
