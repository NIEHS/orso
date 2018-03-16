import json
import os
import re
import requests
from collections import defaultdict

from django.conf import settings

ORGANISM_TO_NCBI_ID = {
    'human': 9606,
    'mouse': 10090,
    'fly': 7227,
    'worm': 6239,
}
ASSEMBLY_TO_ORGANISM = {
    'hg19': 'human',
    'GRCh38': 'human',
    'mm9': 'mouse',
    'mm10': 'mouse',
    'dm3': 'fly',
    'dm6': 'fly',
    'ce10': 'worm',
    'ce11': 'worm',
}

ORGANISM_TO_ALIAS_FILE = dict()
for organism, path in [
    ('human', '9606.protein.aliases.v10.5.txt'),
    ('mouse', '10090.protein.aliases.v10.5.txt'),
    ('fly', '7227.protein.aliases.v10.5.txt'),
    ('worm', '6239.protein.aliases.v10.5.txt'),
]:
    ORGANISM_TO_ALIAS_FILE[organism] = \
        os.path.join(settings.STRING_DIR, path)

ORGANISM_TO_LINK_FILE = dict()
for organism, path in [
    ('human', '9606.protein.links.v10.5.txt'),
    ('mouse', '10090.protein.links.v10.5.txt'),
    ('fly', '7227.protein.links.v10.5.txt'),
    ('worm', '6239.protein.links.v10.5.txt'),
]:
    ORGANISM_TO_LINK_FILE[organism] = \
        os.path.join(settings.STRING_DIR, path)


def standardize_gene_name(gene):

    tag_prefixes = ['eGFP-', 'FLAG-', 'HA-']

    for prefix in tag_prefixes:
        gene = re.sub('^{}'.format(prefix), '', gene)

    return gene.lower()


def get_interaction_partners(genes, organism):
    return get_interaction_partners_local(genes, organism)


def get_interaction_partners_local(genes, organism, interaction_threshold=400):

    # Convert to standardized gene names
    standardized_to_input_names = dict()
    for gene in genes:
        standardized_name = standardize_gene_name(gene)
        standardized_to_input_names[standardized_name] = gene
    relevant_aliases = set(standardized_to_input_names.keys())

    # Get aliases
    alias_to_id = dict()
    id_to_aliases = defaultdict(list)
    with open(ORGANISM_TO_ALIAS_FILE[organism]) as f:
        for line in f:
            if not line.startswith('#'):
                _id, _alias = line.strip().split('\t')[:2]
                _alias = standardize_gene_name(_alias)

                if _alias in relevant_aliases:
                    alias_to_id[_alias] = _id
                    id_to_aliases[_id].append(_alias)

    # Get links
    db_partners = defaultdict(list)
    with open(ORGANISM_TO_LINK_FILE[organism]) as f:
        next(f)
        for line in f:
            id_1, id_2, score = line.strip().split()

            if all([
                id_1 in id_to_aliases,
                id_2 in id_to_aliases,
                int(score) >= interaction_threshold,
            ]):
                db_partners[id_1].append(id_2)
                db_partners[id_2].append(id_1)

    # Create output dict
    interaction_partners = defaultdict(set)
    for alias in relevant_aliases:
        input_name = standardized_to_input_names[alias]
        if alias in alias_to_id:
            alias_id = alias_to_id[alias]
            for interaction_partner in db_partners[alias_id]:
                for interacting_alias in id_to_aliases[interaction_partner]:
                    interaction_partners[input_name].add(
                        standardized_to_input_names[interacting_alias])

    return interaction_partners


def get_interaction_partners_network(genes, organism):

    string_api_url = 'https://string-db.org/api'
    output_format = 'json'
    method = 'network'

    standardized_to_input_names = dict()
    for gene in genes:
        standardized_name = standardize_gene_name(gene)
        standardized_to_input_names[standardized_name] = gene

    species = ORGANISM_TO_NCBI_ID[organism]
    standardized_gene_names = standardized_to_input_names.keys()

    request_url = string_api_url + "/" + output_format + "/" + method + "?"
    request_url += "identifiers=%s" % "%0d".join(standardized_gene_names)
    request_url += "&" + "species=" + str(species)

    response = requests.get(request_url)

    try:
        response_json = json.loads(response.text)
    except json.JSONDecodeError:
        print(response.text)
        raise

    interaction_partners = defaultdict(list)

    if 'Error' not in response_json:
        for interaction in response_json:

            partner_1 = interaction['preferredName_A']
            if partner_1 in standardized_to_input_names:
                partner_1 = standardized_to_input_names[partner_1]

            partner_2 = interaction['preferredName_B']
            if partner_2 in standardized_to_input_names:
                partner_2 = standardized_to_input_names[partner_2]

            interaction_partners[partner_1].append(partner_2)
            interaction_partners[partner_2].append(partner_1)

    return interaction_partners


def get_organism_to_interaction_partners_dict(input_dict):
    organism_to_partners_dict = dict()

    organism_to_genes = defaultdict(set)
    for assembly_name, gene_set in input_dict.items():
        organism_to_genes[ASSEMBLY_TO_ORGANISM[assembly_name]] |= gene_set

    for organism, gene_set in organism_to_genes.items():
        organism_to_partners_dict[organism] = \
            get_interaction_partners(gene_set, organism)

    return organism_to_partners_dict
