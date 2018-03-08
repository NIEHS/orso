import json
import os

from celery import group
from celery.decorators import task
from django.conf import settings

from analysis.string_db import get_interaction_partners
from network import models

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
EXPERIMENT_TYPE_TO_RELEVANT_FIELDS = {
    'RNA-PET': ['cell_type'],
    'siRNA knockdown followed by RNA-seq': ['cell_type', 'target'],
    'DNase-seq': ['cell_type'],
    'RNA-seq': ['cell_type'],
    'shRNA knockdown followed by RNA-seq': ['cell_type', 'target'],
    'eCLIP': ['cell_type', 'target'],
    'HiC': ['cell_type'],
    'CRISPR genome editing followed by RNA-seq': ['cell_type', 'target'],
    'single cell isolation followed by RNA-seq': ['cell_type'],
    'RIP-seq': ['cell_type', 'target'],
    'shRNA knockdown followed by RNA-seq': ['cell_type', 'target'],
    'Repli-chip': ['cell_type'],
    'Repli-seq': ['cell_type'],
    'microRNA-seq': ['cell_type'],
    'CRISPRi followed by RNA-seq': ['cell_type', 'target'],
    'genetic modification followed by DNase-seq': ['cell_type', 'target'],
    'FAIRE-seq': ['cell_type'],
    'ChIP-seq': ['target'],
    'RAMPAGE': ['cell_type'],
    'ATAC-seq': ['cell_type'],
    'CAGE': ['cell_type'],
    'whole-genome shotgun bisulfite sequencing': ['cell_type'],
    'ChIA-PET': ['cell_type', 'target'],
    'MNase-seq': ['cell_type'],
    'Other': ['cell_type'],
}
RELEVANT_CATEGORIES = set([
    'adult stem cell',
    'embryonic structure',
    'hematopoietic system',
    'immune system',
    'integument',
    'connective tissue',
    'skeletal system',
    'muscular system',
    'limb',
    'respiratory system',
    'cardiovascular system',
    'urogenital system',
    'gland',
    'viscus',
    'nervous system',
    'head',
    'sense organ',
    'whole organism',
])


def update_metadata_scores():
    datasets = models.Dataset.objects.all()

    tasks = []
    for dataset in datasets:
        tasks.append(update_dataset_metadata_scores.s(dataset.pk))

    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def update_dataset_metadata_scores(dataset_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)

    # Get relevant datasets
    other_datasets = models.Dataset.objects.filter(
        assembly=dataset.assembly,
        experiment__experiment_type=dataset.experiment.experiment_type,
    )

    # Get BRENDA ontology object
    brenda_ont = (models.Ontology.objects.get(name='brenda_tissue_ontology')
                                         .get_ontology_object())

    # Get ENCODE to BRENDA dict
    encode_to_brenda_path = os.path.join(
        settings.ONTOLOGY_DIR, 'encode_to_brenda.json')
    try:
        with open(encode_to_brenda_path) as f:
            encode_cell_type_to_brenda_name = json.load(f)
    except FileNotFoundError:
        print('ENCODE to BRENDA file not found.')
        encode_cell_type_to_brenda_name = dict()

    # Get relevant categories for each cell type
    cell_types = list(set([dataset.experiment.cell_type]) | set(
        other_datasets.values_list('experiment__cell_type', flat=True)))
    cell_type_to_relevant_categories = dict()
    for cell_type in cell_types:

        if cell_type in encode_cell_type_to_brenda_name:
            brenda_term_name = encode_cell_type_to_brenda_name[cell_type]
            for term, name in brenda_ont.term_to_name.items():
                if name == brenda_term_name:
                    brenda_term = term
        else:
            terms = brenda_ont.get_terms(cell_type)
            if terms:
                brenda_term = sorted(terms)[0]
            else:
                brenda_term = None

        if brenda_term:
            parent_set = set([brenda_term]) \
                | brenda_ont.get_all_parents(brenda_term)
            cell_type_to_relevant_categories[cell_type] = \
                set([brenda_ont.term_to_name[term] for term in parent_set]) \
                & RELEVANT_CATEGORIES
        else:
            cell_type_to_relevant_categories[cell_type] = set()

    # Get STRING interaction partners
    genes = list(set(dataset.experiment.target) | set(
        other_datasets.values_list('experiment__target', flat=True)))
    organism = ASSEMBLY_TO_ORGANISM[dataset.assembly.name]
    interaction_partners = get_interaction_partners(genes, organism)

    ds_1 = dataset
    for ds_2 in other_datasets:

        assembly_1 = ds_1.assembly
        exp_type_1 = ds_1.experiment.experiment_type
        target_1 = ds_1.experiment.target
        cell_type_1 = ds_1.experiment.cell_type

        assembly_2 = ds_2.assembly
        exp_type_2 = ds_2.experiment.experiment_type
        target_2 = ds_2.experiment.target
        cell_type_2 = ds_2.experiment.cell_type

        if all([
            ds_1 != ds_2,
            assembly_1 == assembly_2,
            exp_type_1 == exp_type_2,
        ]):

            if exp_type_1.name in EXPERIMENT_TYPE_TO_RELEVANT_FIELDS:
                relevant_fields = \
                    EXPERIMENT_TYPE_TO_RELEVANT_FIELDS[exp_type_1.name]
            else:
                relevant_fields = \
                    EXPERIMENT_TYPE_TO_RELEVANT_FIELDS['Other']

            sim_comparisons = []

            if 'target' in relevant_fields:
                sim_comparisons.append(any([
                    target_1 == target_2,
                    target_1 in interaction_partners[target_2],
                    target_2 in interaction_partners[target_1],
                ]))

            if 'cell_type' in relevant_fields:
                sim_comparisons.append(any([
                    cell_type_1 == cell_type_2,
                    cell_type_to_relevant_categories[cell_type_1] &
                    cell_type_to_relevant_categories[cell_type_2],
                ]))

            if sim_comparisons:
                is_similar = all(sim_comparisons)
            else:
                is_similar = False

            if ds_1.pk < ds_2.pk:
                _ds_1, _ds_2 = ds_1, ds_2
            else:
                _ds_1, _ds_2 = ds_2, ds_1
            models.DatasetMetadataDistance.objects.update_or_create(
                dataset_1=_ds_1,
                dataset_2=_ds_2,
                defaults={
                    'distance': int(is_similar),
                },
            )
