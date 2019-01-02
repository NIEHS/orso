import json
import pandas as pd

from celery.decorators import task
from django.core.exceptions import ValidationError

from network import models
from network.tasks.analysis.string import String
from network.tasks.utils import run_tasks

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
    'whole organism',  # Added to catch fly and worm samples
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


def compare(obj_1, obj_2, string_obj, check_identity):

    if obj_1['pk'] == obj_2['pk']:
        return True

    if not all([
        obj_1['organism'] == obj_2['organism'],
        obj_1['assemblies'] & obj_2['assemblies'],
        obj_1['experiment_type'] == obj_2['experiment_type'],
    ]):
        return False

    if check_identity and all([
        obj_1['cell_type'] == obj_2['cell_type'],
        obj_1['target'] == obj_2['target'],
    ]):
        return True

    compare_cell_type_classes = \
        bool(obj_1['cell_type_classes'] & obj_2['cell_type_classes'])
    compare_epi_classes = \
        bool(obj_1['target_classes'] & obj_2['target_classes'] &
             set(RELEVANT_EPI_CATEGORIES))
    compare_gene_classes = \
        bool(obj_1['target_classes'] & obj_2['target_classes'] &
             set(RELEVANT_GO_CATEGORIES))
    compare_interaction = string_obj.compare_gene_symbols(
        obj_1['target'], obj_2['target'], obj_1['organism'])
    empty_targets = not any([obj_1['target'], obj_2['target']])

    if all([
        compare_cell_type_classes,
        any([
            compare_epi_classes,
            compare_gene_classes and compare_interaction,
            empty_targets,
        ])
    ]):
        return True
    else:
        return False


def get_similarity_matrix(obj_list, string_obj, check_identity=True):
    d = {}
    for obj_1 in obj_list:
        d.update({obj_1['pk']: pd.Series(
            [compare(obj_1, obj_2, string_obj, check_identity)
             for obj_2 in obj_list],
            index=[obj_2['pk'] for obj_2 in obj_list],
        )})
    return pd.DataFrame(d)


def get_metadata_classes_list(metadata_field_list, ontologies,
                              relevant_field_sets, **kwargs):
    classes_list = []
    for field in metadata_field_list:
        classes_list.append(get_metadata_classes(
            field, ontologies, relevant_field_sets, **kwargs))
    return classes_list


def get_metadata_classes(field, ontologies, relevant_field_sets,
                         term_lookups=None):
    if not term_lookups:
        term_lookups = [None] * len(ontologies)
    classes = set()
    for ont, relevant_fields, term_lookup in \
            zip(ontologies, relevant_field_sets, term_lookups):
        try:
            field = term_lookup[field]
        except (TypeError, KeyError):
            pass
        terms = set(ont.get_terms(field))
        terms.update(ont.get_all_parents(terms))
        term_names = [ont.term_to_name[term] for term in terms]
        classes |= set(term_names) & relevant_fields
    return classes


def get_cell_type_classes_list(cell_type_list):
    brenda_ont = models.Ontology.objects.get(
        name='brenda_tissue_ontology').get_ontology_object()
    brenda_relevant_fields = set(RELEVANT_CELL_TYPE_CATEGORIES)

    encode_lookup = json.load(open('data/ontologies/encode_to_brenda.json'))
    ihec_lookup = json.load(open('data/ontologies/ihec_to_brenda.json'))

    return get_metadata_classes_list(
        cell_type_list,
        [brenda_ont],
        [brenda_relevant_fields],
        term_lookups=[encode_lookup, ihec_lookup],
    )


def get_target_classes_list(target_list):
    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()
    epi_ont = models.Ontology.objects.get(
        name='epigenetic_modification_ontology').get_ontology_object()

    gene_relevant_fields = set(RELEVANT_GO_CATEGORIES)
    epi_relevant_fields = set(RELEVANT_EPI_CATEGORIES)

    return get_metadata_classes_list(
        target_list,
        [gene_ont, epi_ont],
        [gene_relevant_fields, epi_relevant_fields],
    )


def get_experiment_metadata_sim_matrix(exp_list):

    cell_type_classes_list = \
        get_cell_type_classes_list([exp.cell_type for exp in exp_list])
    target_classes_list = \
        get_target_classes_list([exp.target for exp in exp_list])

    string_obj = String()

    obj_list = []

    for exp, cell_type_classes, target_classes in zip(
            exp_list, cell_type_classes_list, target_classes_list):
        obj_list.append({
            'pk': exp.pk,
            'assemblies': set(models.Assembly.objects.filter(
                dataset__experiment=exp)),
            'organism': exp.organism.name,
            'experiment_type': exp.experiment_type,
            'cell_type': exp.cell_type,
            'target': exp.target,
            'cell_type_classes': cell_type_classes,
            'target_classes': target_classes,
        })

    return get_similarity_matrix(obj_list, string_obj, check_identity=True)


def get_dataset_predicted_sim_matrix(ds_list):

    string_obj = String()

    obj_list = []

    for ds in ds_list:
        obj_list.append({
            'pk': ds.pk,
            'assemblies': set([ds.assembly]),
            'organism': ds.experiment.organism.name,
            'experiment_type': ds.experiment.experiment_type,
            'cell_type': ds.experiment.cell_type,
            'target': ds.experiment.target,
            'cell_type_classes': ds.get_predicted_cell_types(),
            'target_classes': ds.get_predicted_targets(),
        })

    return get_similarity_matrix(obj_list, string_obj, check_identity=False)


def update_similarities(objects, sims_df, update_task, **kwargs):
    tasks = []
    for obj in objects:
        row = sims_df[obj.pk]
        other_obj_pks = list([int(val) for val in row.index])
        sims = list([bool(val) for val in row])
        tasks.append(update_task.si(obj.pk, other_obj_pks, sims))
    run_tasks(tasks, **kwargs)


def update_experiment_metadata_similarities(experiments, **kwargs):
    organisms = set([exp.organism for exp in experiments])
    experiment_types = set([exp.experiment_type for exp in experiments])
    total_experiments = list(models.Experiment.objects.filter(
        organism__in=organisms,
        experiment_type__in=experiment_types,
    ))

    sims_df = get_experiment_metadata_sim_matrix(total_experiments)

    update_similarities(experiments, sims_df,
                        update_experiment_metadata_similarity, **kwargs)


def update_dataset_predicted_similarities(datasets, **kwargs):
    assemblies = set([ds.assembly for ds in datasets])
    experiment_types = set([ds.experiment.experiment_type for ds in datasets])
    total_datasets = list(models.Dataset.objects.filter(
        assembly__in=assemblies,
        experiment__experiment_type__in=experiment_types,
    ))

    sims_df = get_dataset_predicted_sim_matrix(total_datasets)

    update_similarities(datasets, sims_df, update_predicted_dataset_similarity,
                        **kwargs)


def update_objects(exp_1, exp_2, sim_type, sim, ds_1=None, ds_2=None):
    if sim:
        try:
            models.Similarity.objects.update_or_create(
                experiment_1=exp_1,
                experiment_2=exp_2,
                dataset_1=ds_1,
                dataset_2=ds_2,
                sim_type=sim_type,
            )
        except ValidationError:
            pass
    else:
        models.Similarity.objects.filter(
            experiment_1=exp_1,
            experiment_2=exp_2,
            dataset_1=ds_1,
            dataset_2=ds_2,
            sim_type=sim_type,
        ).delete()


@task
def update_experiment_metadata_similarity(exp_pk, other_exp_pks, sims):
    exp_1 = models.Experiment.objects.get(pk=exp_pk)
    for pk, sim in zip(other_exp_pks, sims):
        exp_2 = models.Experiment.objects.get(pk=pk)
        if exp_1 != exp_2:
            update_objects(exp_1, exp_2, 'metadata', sim)
            update_objects(exp_2, exp_1, 'metadata', sim)


@task
def update_predicted_dataset_similarity(ds_pk, other_ds_pks, sims):
    ds_1 = models.Dataset.objects.get(pk=ds_pk)
    for pk, sim in zip(other_ds_pks, sims):
        ds_2 = models.Dataset.objects.get(pk=pk)
        if all([
            ds_1 != ds_2,
            ds_1.experiment != ds_2.experiment,
        ]):
            update_objects(ds_1.experiment, ds_2.experiment, 'primary', sim,
                           ds_1=ds_1, ds_2=ds_2)
            update_objects(ds_2.experiment, ds_1.experiment, 'primary', sim,
                           ds_1=ds_1, ds_2=ds_2)
