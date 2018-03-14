import json
import os

import pandas as pd
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
    genes = set([exp.target for exp in experiments])
    assembly = \
        models.Assembly.objects.filter(dataset__experiment__in=experiments)[0]
    organism = ASSEMBLY_TO_ORGANISM[assembly.name]
    interaction_partners = get_interaction_partners(genes, organism)

    d = {}
    exp_list = list(experiments)
    for exp_1 in exp_list:
        comp_values = []
        for exp_2 in exp_list:
            if all([
                set(models.Assembly.objects.filter(dataset__experiment=exp_1)) &  # noqa
                set(models.Assembly.objects.filter(dataset__experiment=exp_2)),
                exp_1.experiment_type ==
                exp_2.experiment_type
            ]):
                if exp_1 == exp_2:
                    comp_values.append(True)
                else:
                    target_1 = exp_1.target
                    target_2 = exp_2.target

                    cell_type_1 = exp_1.cell_type
                    cell_type_2 = exp_2.cell_type

                    exp_type = exp_1.experiment_type.name

                    if exp_type in EXPERIMENT_TYPE_TO_RELEVANT_FIELDS:
                        relevant_fields = \
                            EXPERIMENT_TYPE_TO_RELEVANT_FIELDS[exp_type]
                    else:
                        relevant_fields = \
                            EXPERIMENT_TYPE_TO_RELEVANT_FIELDS['Other']

                    sim_comparisons = []

                    if 'target' in relevant_fields:
                        sim_comparisons.append(any([
                            target_1 == target_2,
                            target_1 in interaction_partners[target_2],
                            target_1 in interaction_partners[target_2],
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


def update_all_metadata_recommendations():
    experiments = models.Experiment.objects.all()
    sims_df = generate_metadata_sims_df(experiments)

    owned_experiments = models.Experiment.objects.filter(owners=True)
    for exp_1 in owned_experiments:
        owners = models.MyUser.objects.filter(experiment=exp_1)
        for exp_2 in experiments:
            if exp_1 != exp_2:
                for owner in owners:
                    if sims_df[exp_1.pk][exp_2.pk]:
                        models.MetadataRec.objects.update_or_create(
                            user=owner,
                            experiment=exp_2,
                            personal_experiment=exp_1,
                        )
                    else:
                        try:
                            models.MetadataRec.objects.get(
                                user=owner,
                                experiment=exp_2,
                                personal_experiment=exp_1,
                            ).delete()
                        except models.MetadataRec.DoesNotExist:
                            pass


@task
def update_metadata_recommendations(experiment_pk):
    experiment = models.Experiment.objects.get(pk=experiment_pk)
    owners = models.MyUser.objects.filter(experiment=experiment)

    # Get relevant experiments
    assemblies = models.Assembly.objects.filter(dataset__experiment=experiment)
    other_experiments = models.Experiment.objects.filter(
        dataset__assembly__in=assemblies,
        experiment_type=experiment.experiment_type,
    )
    total_experiments = set([experiment]) | set(other_experiments)

    sims_df = generate_metadata_sims_df(total_experiments)

    exp_1 = experiment
    for exp_2 in other_experiments:
        if exp_1 != exp_2:
            for owner in owners:
                if sims_df[exp_1.pk][exp_2.pk]:
                    models.MetadataRec.objects.update_or_create(
                        user=owner,
                        experiment=exp_2,
                        personal_experiment=exp_1,
                    )
                else:
                    try:
                        models.MetadataRec.objects.get(
                            user=owner,
                            experiment=exp_2,
                            personal_experiment=exp_1,
                        ).delete()
                    except models.MetadataRec.DoesNotExist:
                        pass
