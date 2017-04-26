#!/usr/bin/env python

import os
import json
import click
import pyBigWig


#  TODO: add to utilities
def check_bigwigs(single_bw, paired_1_bw, paired_2_bw):
    if not single_bw and not paired_1_bw and not paired_2_bw:
        raise ValueError('No bigWigs specified.')
    if single_bw and (paired_1_bw or paired_2_bw):
        raise ValueError('A single bigWig of ambiguous strand may not be run with paired bigWigs.')  # noqa
    if not single_bw and not (paired_1_bw and paired_2_bw):
        raise ValueError('If a single bigWig of ambiguous strand is not used, paired bigWigs are required.')  # noqa

    if single_bw:
        try:
            pyBigWig.open(single_bw)
        except RuntimeError:
            print('pyBigWig is unable to open bigWig file.')
            return False
    else:
        try:
            pyBigWig.open(paired_1_bw)
            pyBigWig.open(paired_2_bw)
        except RuntimeError:
            print('pyBigWig is unable to open bigWig files.')
            return False

    return True


class GeneCoverage(object):

    def __init__(self, bed_file, single_bw=None, paired_1_bw=None,
                 paired_2_bw=None):

        self.bed_file = bed_file

        self.single_bw = single_bw
        self.paired_1_bw = paired_1_bw
        self.paired_2_bw = paired_2_bw

        assert os.path.exists(self.bed_file)

        if check_bigwigs(self.single_bw, self.paired_1_bw, self.paired_2_bw):
            self.stranded = self.check_stranded()
            self.genes = self.read_genes_from_bed_file(
                self.bed_file)
            self.find_coverage_values()

    def check_stranded(self):
        if self.paired_1_bw and self.paired_2_bw:
            return True
        elif self.single_bw:
            return False
        else:
            raise ValueError('Improper bigWigs selected: strand inconsistentcy')  # noqa

    @staticmethod
    def read_genes_from_bed_file(bed_file):
        genes = []

        with open(bed_file) as f:
            for line in f:
                if line[0] == '#':  # Check for header
                    pass
                else:
                    chromosome, start, end, common_name, zero, strand = \
                        line.strip().split()
                    genes.append(
                        (common_name, chromosome, strand, start, end)
                    )

        return genes

    def find_coverage_values(self):

        def get_coverage(pbw_obj, chromosome, start, end):
            try:
                coverage = pbw_obj.stats(chromosome, int(start), int(end))[0]
            except RuntimeError:
                coverage = None

            if not coverage:
                return 0
            else:
                return coverage

        self.coverage_values = []

        if self.single_bw:
            single_bw = pyBigWig.open(self.single_bw)
        if self.paired_1_bw:
            paired_1_bw = pyBigWig.open(self.paired_1_bw)
        if self.paired_1_bw:
            paired_2_bw = pyBigWig.open(self.paired_2_bw)

        for gene in self.genes:
            common_name, chromosome, strand, gene_start, gene_end = gene
            coverage = 0

            if self.single_bw:
                coverage += get_coverage(
                    single_bw, chromosome, gene_start, gene_end)
            if self.paired_1_bw and strand == '+':
                coverage += get_coverage(
                    paired_1_bw, chromosome, gene_start, gene_end)
            if self.paired_2_bw and strand == '-':
                coverage += get_coverage(
                    paired_2_bw, chromosome, gene_start, gene_end)

            self.coverage_values.append(coverage)

    def print_coverage_values(self, output_json):
        with open(output_json, 'w') as OUT:
            json.dump(self.coverage_values, OUT)


@click.command()
@click.argument('bed_file')
@click.argument('output_json')
@click.option('-s', help='Single bigWig (Ambiguous strand)', type=str)
@click.option('-f', help='Forward strand (+) bigWig', type=str)
@click.option('-r', help='Reverse strand (-) bigWig', type=str)
def cli(bed_file, output_json, s, f, r):
    '''
    Find coverage values over genes in the GTF file
    '''

    gene_coverage = GeneCoverage(bed_file, s, f, r,)

    try:
        gene_coverage.print_coverage_values(output_json)
    except:
        print('Unable to print gene coverage values.')

if __name__ == '__main__':
    cli()
