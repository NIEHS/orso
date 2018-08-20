import json
import os
from collections import defaultdict

import pandas as pd
from celery import group
from celery.decorators import task
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db.models import Q

from network.tasks.analysis.string_db import (
    ASSEMBLY_TO_ORGANISM, get_organism_to_interaction_partners_dict)
from network import models
from network.tasks.recommendations import update_recommendations

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
RELEVANT_CELL_TYPE_CATEGORIES = [
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
]
RELEVANT_EPI_CATEGORIES = [
    'At active promoters',
    'At inactive promoters',
    'At poised promoters',
    'At active enhancers',
    'At inactive enhancers',
    'At active genes',
    'At inactive genes',
]
RELEVANT_GO_CATEGORIES = [
    'positive regulation of transcription from RNA polymerase II promoter',
    'negative regulation of transcription from RNA polymerase II promoter',
]


def generate_metadata_sims_df(experiments, identity_only=False):

    relevant_cell_type_set = set(RELEVANT_CELL_TYPE_CATEGORIES)
    relevant_epi_set = set(RELEVANT_EPI_CATEGORIES)
    relevant_go_set = set(RELEVANT_GO_CATEGORIES)

    # Get ontology objects
    brenda_ont = models.Ontology.objects.get(
        name='brenda_tissue_ontology').get_ontology_object()
    epi_ont = models.Ontology.objects.get(
        name='epigenetic_modification_ontology').get_ontology_object()
    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()

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
    cell_types = set([exp.cell_type for exp in experiments])
    cell_type_to_relevant_categories = dict()
    for cell_type in cell_types:

        brenda_term = None
        if cell_type in encode_cell_type_to_brenda_name:
            brenda_term_name = encode_cell_type_to_brenda_name[cell_type]
            for term, name in brenda_ont.term_to_name.items():
                if name == brenda_term_name:
                    brenda_term = term
        else:
            terms = brenda_ont.get_terms(cell_type)
            if terms:
                brenda_term = sorted(terms)[0]

        if brenda_term:
            parent_set = set([brenda_term]) \
                | brenda_ont.get_all_parents(brenda_term)
            cell_type_to_relevant_categories[cell_type] = \
                set([brenda_ont.term_to_name[term] for term in parent_set]) \
                & relevant_cell_type_set
        else:
            cell_type_to_relevant_categories[cell_type] = set()

    # Get relevant categories for each target
    targets = set([exp.target for exp in experiments])
    target_to_relevant_categories = dict()
    for target in targets:

        categories = set()

        go_terms = gene_ont.get_terms(target)
        epi_terms = epi_ont.get_terms(target)

        if go_terms:

            parent_set = set(go_terms)
            for term in go_terms:
                parent_set.update(set(gene_ont.get_all_parents(term)))
            categories.update(
                set([gene_ont.term_to_name[term] for term in parent_set]) &
                relevant_go_set
            )

        if epi_terms:

            parent_set = set(epi_terms)
            for term in epi_terms:
                parent_set.update(set(epi_ont.get_all_parents(term)))
            categories.update(
                set([epi_ont.term_to_name[term] for term in parent_set]) &
                relevant_epi_set
            )

        target_to_relevant_categories[target] = categories

    # Get experiment to assemblies
    experiment_to_assemblies = dict()
    for exp in experiments:
        experiment_to_assemblies[exp] = set(
            models.Assembly.objects.filter(dataset__experiment=exp))

    # Get STRING interaction partners
    gene_dict = defaultdict(set)
    for exp in experiments:
        for ds in models.Dataset.objects.filter(experiment=exp):
            gene_dict[ds.assembly.name].add(exp.target)
    interaction_partners = get_organism_to_interaction_partners_dict(gene_dict)

    d = {}
    exp_list = list(experiments)
    for exp_1 in exp_list:
        comp_values = []
        for exp_2 in exp_list:
            if all([
                experiment_to_assemblies[exp_1] &
                experiment_to_assemblies[exp_2],
                exp_1.experiment_type ==
                exp_2.experiment_type,
            ]):
                if exp_1 == exp_2:
                    comp_values.append(True)
                else:
                    target_1 = exp_1.target
                    target_2 = exp_2.target

                    cell_type_1 = exp_1.cell_type
                    cell_type_2 = exp_2.cell_type

                    sim_comparisons = []

                    if target_1 or target_2:

                        is_tf = bool(
                            relevant_go_set &
                            target_to_relevant_categories[target_1] &
                            target_to_relevant_categories[target_2]
                        )
                        is_hist = bool(
                            relevant_epi_set &
                            target_to_relevant_categories[target_1] &
                            target_to_relevant_categories[target_2]
                        )

                        # Check if interacting STRING
                        organism = ASSEMBLY_TO_ORGANISM[
                            list(experiment_to_assemblies[exp_1])[0].name]
                        _interaction_partners = interaction_partners[organism]
                        interacting = any([
                            target_1 in _interaction_partners[target_2],
                            target_2 in _interaction_partners[target_1],
                        ])

                        if identity_only:
                            sim_comparisons.append(target_1 == target_2)
                        else:
                            sim_comparisons.append(any([
                                target_1 == target_2,
                                (is_tf and interacting),
                                is_hist,
                            ]))

                    if cell_type_1 or cell_type_2:
                        if identity_only:
                            sim_comparisons.append(cell_type_1 == cell_type_2)
                        else:
                            sim_comparisons.append(any([
                                cell_type_1 == cell_type_2,
                                cell_type_to_relevant_categories[cell_type_1] &
                                cell_type_to_relevant_categories[cell_type_2],
                            ]))

                    if sim_comparisons:
                        is_similar = all(sim_comparisons)
                    else:
                        is_similar = False

                    comp_values.append(is_similar)
            else:
                comp_values.append(False)

        series = pd.Series(
            comp_values, index=[exp.pk for exp in exp_list])
        d.update({exp_1.pk: series})

    return pd.DataFrame(d)


def generate_metadata_sims_df_for_datasets(datasets, identity_only=False):
    experiments = \
        models.Experiment.objects.filter(dataset__in=datasets).distinct()
    exp_sims = generate_metadata_sims_df(experiments,
                                         identity_only=identity_only)

    ds_pks = [ds.pk for ds in datasets]
    ds_sims = pd.DataFrame(index=[ds_pks], columns=[ds_pks])
    for ds_1 in datasets:
        for ds_2 in datasets:
            ds_sims.loc[ds_1.pk, ds_2.pk] = \
                exp_sims[ds_1.experiment.pk][ds_2.experiment.pk]

    return ds_sims


def update_all_metadata_sims_and_recs():
    exp_pks = list([exp.pk for exp in models.Experiment.objects.all()])
    update_bulk_similarities(exp_pks)

    tasks = []
    user_experiments = models.Experiment.objects.filter(
        Q(owners=True) | Q(favorite__user=True))
    for experiment in user_experiments:
        tasks.append(update_recommendations.si(
            experiment.pk, sim_types=['metadata']))

    job = group(tasks)
    results = job.apply_async()
    results.join()


# Designed to update similarities for many experiments at once. Uses celery
# to query and update similarity objects.
def update_bulk_similarities(experiment_pks):
    experiments = models.Experiment.objects.filter(pk__in=experiment_pks)
    df = generate_metadata_sims_df(experiments)

    tasks = []
    for exp_pk in experiment_pks:
        row = df[exp_pk]
        other_exp_pks = list([int(val) for val in row.index])
        sims = list([bool(val) for val in row])
        tasks.append(_update_similarity.si(exp_pk, other_exp_pks, sims))
    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def _update_similarity(exp_pk, other_exp_pks, sims):
    exp_1 = models.Experiment.objects.get(pk=exp_pk)
    for pk, sim in zip(other_exp_pks, sims):
        exp_2 = models.Experiment.objects.get(pk=pk)
        if exp_1 != exp_2:
            if sim:
                try:
                    models.Similarity.objects.update_or_create(
                        experiment_1=exp_1,
                        experiment_2=exp_2,
                        sim_type='metadata',
                    )
                except ValidationError:
                    pass

                try:
                    models.Similarity.objects.update_or_create(
                        experiment_1=exp_2,
                        experiment_2=exp_1,
                        sim_type='metadata',
                    )
                except ValidationError:
                    pass
            else:
                models.Similarity.objects.filter(
                    experiment_1=exp_1,
                    experiment_2=exp_2,
                    sim_type='metadata',
                ).delete()
                models.Similarity.objects.filter(
                    experiment_1=exp_2,
                    experiment_2=exp_1,
                    sim_type='metadata',
                ).delete()


@task
def update_metadata_sims_and_recs(experiment_pk):
    update_metadata_similarities([experiment_pk])
    update_recommendations(experiment_pk, sim_types=['metadata'], lock=False)


@task
def update_metadata_similarities(experiment_pks):
    experiments = models.Experiment.objects.filter(pk__in=experiment_pks)

    # Get relevant experiments
    organisms = models.Organism.objects.filter(
        assembly__dataset__experiment__in=experiments)
    experiment_types = models.ExperimentType.objects.filter(
        experiment__in=experiments)
    other_experiments = models.Experiment.objects.filter(
        dataset__assembly__organism__in=organisms,
        experiment_type__in=experiment_types,
    )
    total_experiments = set(experiments) | set(other_experiments)

    # Get experiment to organism
    experiment_to_organism = dict()
    for exp in total_experiments:
        experiment_to_organism[exp] = set(
            models.Organism.objects.filter(assembly__dataset__experiment=exp))

    sims_df = generate_metadata_sims_df(total_experiments)

    for exp_1 in experiments:
        for exp_2 in other_experiments:
            if all([
                exp_1 != exp_2,
                exp_1.experiment_type == exp_2.experiment_type,
                experiment_to_organism[exp_1] &
                experiment_to_organism[exp_2],
            ]):
                    if sims_df[exp_1.pk][exp_2.pk]:
                        try:
                            models.Similarity.objects.update_or_create(
                                experiment_1=exp_1,
                                experiment_2=exp_2,
                                sim_type='metadata',
                            )
                        except ValidationError:
                            pass

                        try:
                            models.Similarity.objects.update_or_create(
                                experiment_1=exp_2,
                                experiment_2=exp_1,
                                sim_type='metadata',
                            )
                        except ValidationError:
                            pass
                    else:
                        models.Similarity.objects.filter(
                            experiment_1=exp_1,
                            experiment_2=exp_2,
                            sim_type='metadata',
                        ).delete()
                        models.Similarity.objects.filter(
                            experiment_1=exp_2,
                            experiment_2=exp_1,
                            sim_type='metadata',
                        ).delete()
