import csv
import json
import os
import re
from collections import defaultdict

from django.conf import settings

alias_json = os.path.join(settings.STRING_DIR, 'alias.json')
interaction_json = os.path.join(settings.STRING_DIR, 'interaction.json')

ORGANISM_TO_ALIAS_FILE = dict()
for organism, path in [
    ('Homo sapiens', '9606.protein.aliases.v10.5.txt'),
    ('Mus musculus', '10090.protein.aliases.v10.5.txt'),
    ('Drosophila melanogaster', '7227.protein.aliases.v10.5.txt'),
    ('Caenorhabditis elegans', '6239.protein.aliases.v10.5.txt'),
]:
    ORGANISM_TO_ALIAS_FILE[organism] = \
        os.path.join(settings.STRING_DIR, path)

ORGANISM_TO_LINK_FILE = dict()
for organism, path in [
    ('Homo sapiens', '9606.protein.links.v10.5.txt'),
    ('Mus musculus', '10090.protein.links.v10.5.txt'),
    ('Drosophila melanogaster', '7227.protein.links.v10.5.txt'),
    ('Caenorhabditis elegans', '6239.protein.links.v10.5.txt'),
]:
    ORGANISM_TO_LINK_FILE[organism] = \
        os.path.join(settings.STRING_DIR, path)


def standardize_gene_name(gene):

    tag_prefixes = ['eGFP-', 'FLAG-', 'HA-']

    for prefix in tag_prefixes:
        gene = re.sub('^{}'.format(prefix), '', gene)

    return gene.lower()


class String(object):

    def __init__(self):

        self.alias_to_gene = self.get_alias_to_gene()
        self.interaction_partners = self.get_interaction_partners()

    def get_alias_to_gene(self):
        if os.path.isfile(alias_json):
            return json.load(open(alias_json))
        return self.create_alias_to_gene()

    def create_alias_to_gene(self):
        alias_to_gene = dict()
        for organism, alias_path in ORGANISM_TO_ALIAS_FILE.items():
            alias_to_gene[organism] = defaultdict(set)
            with open(alias_path) as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader)
                for row in reader:
                    alias_to_gene[organism][
                        standardize_gene_name(row[1])].add(row[0])
            for key, value in alias_to_gene[organism].items():
                alias_to_gene[organism][key] = list(value)
        return alias_to_gene

    def write_alias_json(self):
        json.dump(self.alias_to_gene, open(alias_json, 'w'))

    def get_interaction_partners(self):
        if os.path.isfile(interaction_json):
            return json.load(open(interaction_json))
        return self.create_interaction_partners()

    def create_interaction_partners(self):
        interaction_partners = dict()
        for organism, interaction_path in ORGANISM_TO_LINK_FILE.items():
            interaction_partners[organism] = defaultdict(set)
            with open(interaction_path) as f:
                reader = csv.reader(f, delimiter=' ')
                next(reader)
                for row in reader:
                    partner_1, partner_2, score = row
                    if int(score) >= 400:
                        interaction_partners[organism][
                            partner_1].add(partner_2)
            for key, value in interaction_partners[organism].items():
                interaction_partners[organism][key] = list(value)
        return interaction_partners

    def write_interaction_json(self):
        json.dump(self.interaction_partners, open(interaction_json, 'w'))

    def compare_gene_symbols(self, alias_1, alias_2, organism):

        def compare(genes_1, genes_2):
            for gene in genes_1:
                try:
                    if set(genes_2) & \
                            set(self.interaction_partners[organism][gene]):
                        return True
                except KeyError:
                    pass
            return False

        alias_1 = standardize_gene_name(alias_1)
        alias_2 = standardize_gene_name(alias_2)
        try:
            genes_1 = self.alias_to_gene[organism][alias_1]
            genes_2 = self.alias_to_gene[organism][alias_2]
        except KeyError:
            return False

        return any([
            compare(genes_1, genes_2),
            compare(genes_2, genes_1),
        ])
