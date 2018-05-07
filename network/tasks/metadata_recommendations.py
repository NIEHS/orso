import json
import os
from collections import defaultdict

import pandas as pd
from celery import group
from celery.decorators import task
from django.conf import settings
from django.db.models import Q

from analysis.string_db import (
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


def generate_metadata_sims_df(experiments):

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
    cell_types = set([exp.cell_type for exp in experiments])
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
    gene_dict = defaultdict(set)
    for exp in experiments:
        for ds in models.Dataset.objects.filter(experiment=exp):
            gene_dict[ds.assembly.name].add(exp.target)
    interaction_partners = get_organism_to_interaction_partners_dict(gene_dict)

    # Get experiment to assemblies
    experiment_to_assemblies = dict()
    for exp in experiments:
        experiment_to_assemblies[exp] = set(
            models.Assembly.objects.filter(dataset__experiment=exp))

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

                    exp_type = exp_1.experiment_type.name
                    _assembly = list(experiment_to_assemblies[exp_1])[0]
                    organism = ASSEMBLY_TO_ORGANISM[_assembly.name]

                    if exp_type in EXPERIMENT_TYPE_TO_RELEVANT_FIELDS:
                        relevant_fields = \
                            EXPERIMENT_TYPE_TO_RELEVANT_FIELDS[exp_type]
                    else:
                        relevant_fields = \
                            EXPERIMENT_TYPE_TO_RELEVANT_FIELDS['Other']

                    sim_comparisons = []
                    _interaction_partners = interaction_partners[organism]

                    if 'target' in relevant_fields:
                        sim_comparisons.append(any([
                            target_1 == target_2,
                            target_1 in _interaction_partners[target_2],
                            target_1 in _interaction_partners[target_2],
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

                    comp_values.append(is_similar)
            else:
                comp_values.append(False)

        series = pd.Series(
            comp_values, index=[exp.pk for exp in exp_list])
        d.update({exp_1.pk: series})

    return pd.DataFrame(d)


def generate_metadata_sims_df_for_datasets(datasets):
    experiments = \
        models.Experiment.objects.filter(dataset__in=datasets).distinct()
    exp_sims = generate_metadata_sims_df(experiments)

    ds_pks = [ds.pk for ds in datasets]
    ds_sims = pd.DataFrame(index=[ds_pks], columns=[ds_pks])
    for ds_1 in datasets:
        for ds_2 in datasets:
            ds_sims.loc[ds_1.pk, ds_2.pk] = \
                exp_sims[ds_1.experiment.pk][ds_2.experiment.pk]

    return ds_sims


def update_all_metadata_sims_and_recs():
    all_experiments = models.Experiment.objects.all()
    update_metadata_similarities([exp.pk for exp in all_experiments])

    user_experiments = models.Experiment.objects.filter(
        Q(owners=True) | Q(favorite__user=True))

    tasks = []
    for experiment in user_experiments:
        tasks.append(update_recommendations.si(
            experiment.pk, sim_types=['metadata']))

    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def update_metadata_sims_and_recs(experiment_pk):
    update_metadata_similarities([experiment_pk])
    update_recommendations(experiment_pk, sim_types=['metadata'], lock=False)


@task
def update_metadata_similarities(experiment_pks):
    experiments = models.Experiment.objects.filter(pk__in=experiment_pks)

    # Get relevant experiments
    assemblies = models.Assembly.objects.filter(
        dataset__experiment__in=experiments)
    experiment_types = models.ExperimentType.objects.filter(
        experiment__in=experiments)
    other_experiments = models.Experiment.objects.filter(
        dataset__assembly__in=assemblies,
        experiment_type__in=experiment_types,
    )
    total_experiments = set(experiments) | set(other_experiments)

    # Get experiment to assemblies
    experiment_to_assemblies = dict()
    for exp in total_experiments:
        experiment_to_assemblies[exp] = set(
            models.Assembly.objects.filter(dataset__experiment=exp))

    sims_df = generate_metadata_sims_df(total_experiments)

    for exp_1 in experiments:
        for exp_2 in other_experiments:
            if all([
                exp_1 != exp_2,
                exp_1.experiment_type == exp_2.experiment_type,
                experiment_to_assemblies[exp_1] &
                experiment_to_assemblies[exp_2],
            ]):
                    if sims_df[exp_1.pk][exp_2.pk]:
                        models.Similarity.objects.update_or_create(
                            experiment_1=exp_1,
                            experiment_2=exp_2,
                            sim_type='metadata',
                        )
                        models.Similarity.objects.update_or_create(
                            experiment_1=exp_2,
                            experiment_2=exp_1,
                            sim_type='metadata',
                        )
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
