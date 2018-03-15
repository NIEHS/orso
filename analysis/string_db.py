import json
import re
import requests
from collections import defaultdict

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


def standardize_gene_name(gene):

    tag_prefixes = ['eGFP-', 'FLAG-', 'HA-']

    for prefix in tag_prefixes:
        gene = re.sub('^{}'.format(prefix), '', gene)

    return gene


def get_interaction_partners(genes, organism):

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
    response_json = json.loads(response.text)

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
