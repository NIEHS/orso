import json
import os

from celery.decorators import task
from django.conf import settings
from progress.bar import Bar

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
    update_dataset_metadata_scores(datasets)


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

            relevant_fields = \
                EXPERIMENT_TYPE_TO_RELEVANT_FIELDS[exp_type_1.name]

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

            if ds_1.pk > ds_2.pk:
                ds_1, ds_2 = ds_2, ds_1
            models.DatasetMetadataDistance.objects.update_or_create(
                dataset_1=ds_1,
                dataset_2=ds_2,
                defaults={
                    'distance': int(is_similar),
                },
            )


@task
def _update_dataset_metadata_scores(datasets):
    '''
    Update or create dataset metadata distance values.
    '''
    bar = Bar('Processing', max=len(datasets))

    onts = []
    onts.append(models.Ontology.objects.get(name='cell_ontology'))
    onts.append(models.Ontology.objects.get(name='cell_line_ontology'))
    onts.append(models.Ontology.objects.get(name='brenda_tissue_ontology'))
    cell_ont_list = [ont.get_ontology_object() for ont in onts]

    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()

    epi_ont = models.Ontology.objects.get(
        name='epigenetic_modification_ontology').get_ontology_object()

    for ds_1 in datasets:
        for ds_2 in models.Dataset.objects.filter(
            assembly=ds_1.assembly,
            experiment__experiment_type=ds_1.experiment.experiment_type,
        ):

            if ds_1 != ds_2:

                exp_1 = models.Experiment.objects.get(dataset=ds_1)
                exp_2 = models.Experiment.objects.get(dataset=ds_2)

                total_sim = 0

                cell_ont_sims = []
                for ont_obj in cell_ont_list:
                    sim = ont_obj.get_word_similarity(
                        exp_1.cell_type, exp_2.cell_type, metric='lin')
                    if sim:
                        cell_ont_sims.append(sim)
                if cell_ont_sims:
                    total_sim += max(cell_ont_sims)

                gene_ont_sim = gene_ont.get_word_similarity(
                    exp_1.target, exp_2.target, metric='jaccard',
                    weighting='information_content')
                if gene_ont_sim:
                    total_sim += gene_ont_sim

                epi_ont_sim = epi_ont.get_word_similarity(
                    exp_1.target, exp_2.target, metric='lin')
                if epi_ont_sim:
                    total_sim += epi_ont_sim

                models.DatasetMetadataDistance.objects.update_or_create(
                    dataset_1=ds_1,
                    dataset_2=ds_2,
                    defaults={
                        'distance': total_sim,
                    },
                )

        bar.next()

    bar.finish()


@task
def update_experiment_metadata_scores(experiments):
    '''
    Update or create experiment metadata distance values.
    '''
    bar = Bar('Processing', max=len(experiments))

    onts = []
    onts.append(models.Ontology.objects.get(name='cell_ontology'))
    onts.append(models.Ontology.objects.get(name='cell_line_ontology'))
    onts.append(models.Ontology.objects.get(name='brenda_tissue_ontology'))
    cell_ont_list = [ont.get_ontology_object() for ont in onts]

    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()

    epi_ont = models.Ontology.objects.get(
        name='epigenetic_modification_ontology').get_ontology_object()

    for exp_1 in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp_1)
        for exp_2 in models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp_1.experiment_type,
        ):

            if exp_1 != exp_2:

                total_sim = 0

                cell_ont_sims = []
                for ont_obj in cell_ont_list:
                    sim = ont_obj.get_word_similarity(
                        exp_1.cell_type, exp_2.cell_type, metric='lin')
                    if sim:
                        cell_ont_sims.append(sim)
                if cell_ont_sims:
                    total_sim += max(cell_ont_sims)

                gene_ont_sim = gene_ont.get_word_similarity(
                    exp_1.target, exp_2.target, metric='jaccard',
                    weighting='information_content')
                if gene_ont_sim:
                    total_sim += gene_ont_sim

                epi_ont_sim = epi_ont.get_word_similarity(
                    exp_1.target, exp_2.target, metric='lin')
                if epi_ont_sim:
                    total_sim += epi_ont_sim

                models.ExperimentMetadataDistance.objects.update_or_create(
                    experiment_1=exp_1,
                    experiment_2=exp_2,
                    defaults={
                        'distance': total_sim,
                    },
                )

        bar.next()

    bar.finish()
