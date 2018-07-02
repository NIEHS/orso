import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from celery import group
from celery.decorators import task
from django.conf import settings
from django.core.exceptions import ValidationError
from keras import backend as K
from keras import layers
from keras.callbacks import EarlyStopping
from keras.constraints import max_norm
from keras.models import load_model, Sequential
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from analysis.string_db import (
    ASSEMBLY_TO_ORGANISM, get_organism_to_interaction_partners_dict)
from network import models
from network.tasks.metadata_recommendations import \
    EXPERIMENT_TYPE_TO_RELEVANT_FIELDS, RELEVANT_CATEGORIES


def create_neural_networks():

    for lg in models.LocusGroup.objects.all():
        for exp_type in models.ExperimentType.objects.all():
            if models.Dataset.objects.filter(
                assembly=lg.assembly, experiment__experiment_type=exp_type,
                experiment__project__name='ENCODE',
            ).count() >= 100:

                models.NeuralNetwork.objects.get_or_create(
                    locus_group=lg,
                    experiment_type=exp_type,
                    metadata_field='cell_type',
                )
                models.NeuralNetwork.objects.get_or_create(
                    locus_group=lg,
                    experiment_type=exp_type,
                    metadata_field='target',
                )


def fit_neural_networks():

    tasks = []

    for lg in models.LocusGroup.objects.all():
        for exp_type in models.ExperimentType.objects.all():
            if models.Dataset.objects.filter(
                assembly=lg.assembly, experiment__experiment_type=exp_type,
                experiment__project__name='ENCODE',
            ).count() >= 100:

                nn = models.NeuralNetwork.objects.get_or_create(
                    locus_group=lg,
                    experiment_type=exp_type,
                    metadata_field='cell_type',
                )[0]
                tasks.append(fit_neural_network.si(nn.pk))

                nn = models.NeuralNetwork.objects.get_or_create(
                    locus_group=lg,
                    experiment_type=exp_type,
                    metadata_field='target',
                )[0]
                tasks.append(fit_neural_network.si(nn.pk))

    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def fit_neural_network(network_pk):

    network = models.NeuralNetwork.objects.get(pk=network_pk)

    # Get all associated datasets
    datasets = models.Dataset.objects.filter(
        assembly=network.locus_group.assembly,
        experiment__experiment_type=network.experiment_type,
        experiment__project__name='ENCODE',
    )

    # Get x and y arrays
    x = []
    y = []

    for ds in datasets:

        dij = models.DatasetIntersectionJson.objects.get(
            dataset=ds, locus_group=network.locus_group)

        x.append(dij.get_norm_vector())
        y.append(getattr(ds.experiment, network.metadata_field))

    # Generate training and test groups
    z = list(zip(x, y))
    random.shuffle(z)
    x, y = zip(*z)

    index = int(len(x) * .9)
    x_training = x[:index]
    y_training = y[:index]
    x_test = x[index:]
    y_test = y[index:]

    # Fit
    scaler = StandardScaler()
    scaler.fit(x_training)

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(y)

    input_dims = len(x[0])
    if len(label_binarizer.classes_) <= 2:
        output_dims = 1
        loss = 'binary_crossentropy'
    else:
        output_dims = len(label_binarizer.classes_)
        loss = 'categorical_crossentropy'
    unit_count = min(2000, int((input_dims + output_dims) / 2))
    model = _get_model(input_dims, output_dims, unit_count, loss=loss)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.01,
        verbose=0,
        mode='auto',
    )

    history = model.fit(
        scaler.transform(x_training),
        label_binarizer.transform(y_training),
        epochs=2000,
        validation_split=0.11,
        verbose=0,
        callbacks=[early_stopping],
    )
    accuracy = model.evaluate(
        scaler.transform(x_test),
        label_binarizer.transform(y_test),
    )[1]

    # Save
    fn = network.get_nn_model_path()
    model.save(fn)
    network.neural_network_file = fn

    network.neural_network_scaler = scaler
    network.neural_network_label_binarizer = label_binarizer
    network.accuracy = accuracy
    network.training_history = history.history
    network.save()

    if K.backend() == 'tensorflow':
        K.clear_session()


def _get_model(input_dims, output_dims, unit_count,
               loss='categorical_crossentropy'):

    model = Sequential()

    model.add(layers.Dropout(0.2, input_shape=(input_dims,)))
    model.add(layers.Dense(units=unit_count, activation='relu',
              kernel_constraint=max_norm(3)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=unit_count, activation='relu',
              kernel_constraint=max_norm(3)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=output_dims, activation='softmax'))

    sgd = SGD(lr=0.0001)

    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])

    return model


def predict_all_dataset_fields():

    tasks = []
    for ds in models.Dataset.objects.all():
        tasks.append(predict_dataset_fields.si(ds.pk))

    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def predict_dataset_fields(dataset_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)

    dataset.predicted_cell_type = predict_dataset_field(dataset, 'cell_type')
    dataset.predicted_target = predict_dataset_field(dataset, 'target')

    dataset.save()


def predict_dataset_field(dataset, metadata_field):

    try:
        nn = models.NeuralNetwork.objects.filter(
            locus_group__assembly=dataset.assembly,
            experiment_type=dataset.experiment.experiment_type,
            metadata_field=metadata_field,
        ).order_by('-accuracy')[0]

        intersection_vector = models.DatasetIntersectionJson.objects.get(
            dataset=dataset,
            locus_group=nn.locus_group,
        ).get_norm_vector()

    except (
        IndexError,
        AttributeError,
        models.DatasetIntersectionJson.DoesNotExist,
    ):
        return None

    else:
        scaled_vector = nn.neural_network_scaler.transform(intersection_vector)

        model = load_model(nn.neural_network_file.path)

        if len(nn.neural_network_label_binarizer.classes_) == 1:
            predicted_class = nn.neural_network_label_binarizer.classes_[0]
        else:
            index = model.predict_classes(np.array([scaled_vector]))[0]
            predicted_class = nn.neural_network_label_binarizer.classes_[index]

        if not predicted_class:
            predicted_class = None

        if K.backend() == 'tensorflow':
            K.clear_session()

        return predicted_class


def generate_predicted_sims_df(datasets, identity_only=False):

    datasets = list(datasets)

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
    cell_types = set([ds.predicted_cell_type for ds in datasets
                      if ds.predicted_cell_type is not None])
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
                & RELEVANT_CATEGORIES
        else:
            cell_type_to_relevant_categories[cell_type] = set()

    # Get STRING interaction partners
    gene_dict = defaultdict(set)
    for ds in datasets:
        if ds.predicted_target:
            gene_dict[ds.assembly.name].add(ds.predicted_target)
    interaction_partners = get_organism_to_interaction_partners_dict(gene_dict)

    d = {}
    for ds_1 in datasets:
        comp_values = []
        for ds_2 in datasets:
            if all([
                ds_1.assembly == ds_2.assembly,
                ds_1.experiment.experiment_type ==
                ds_2.experiment.experiment_type,
            ]):
                if ds_1 == ds_2:
                    comp_values.append(True)
                else:
                    target_1 = ds_1.predicted_target
                    target_2 = ds_2.predicted_target

                    cell_type_1 = ds_1.predicted_cell_type
                    cell_type_2 = ds_2.predicted_cell_type

                    exp_type = ds_1.experiment.experiment_type.name
                    organism = ASSEMBLY_TO_ORGANISM[ds_1.assembly.name]

                    if exp_type in EXPERIMENT_TYPE_TO_RELEVANT_FIELDS:
                        relevant_fields = \
                            EXPERIMENT_TYPE_TO_RELEVANT_FIELDS[exp_type]
                    else:
                        relevant_fields = \
                            EXPERIMENT_TYPE_TO_RELEVANT_FIELDS['Other']

                    sim_comparisons = []
                    _interaction_partners = interaction_partners[organism]

                    if 'target' in relevant_fields:
                        if target_1 and target_2:
                            sim_comparisons.append(any([
                                target_1 == target_2,
                                target_1 in _interaction_partners[target_2],
                                target_2 in _interaction_partners[target_1],
                            ]))

                    if 'cell_type' in relevant_fields:
                        if cell_type_1 and cell_type_2:
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
            comp_values, index=[ds.pk for ds in datasets])
        d.update({ds_1.pk: series})

    return pd.DataFrame(d)


def update_all_similarities():
    datasets = models.Dataset.objects.all()
    update_bulk_similarities([ds.pk for ds in datasets])


def update_bulk_similarities(dataset_pks):
    datasets = models.Dataset.objects.filter(pk__in=dataset_pks)
    df = generate_predicted_sims_df(datasets)

    tasks = []
    for ds_pk in dataset_pks:
        row = df[ds_pk]
        other_ds_pks = list([int(val) for val in row.index])
        sims = list([bool(val) for val in row])
        tasks.append(_update_similarity.si(ds_pk, other_ds_pks, sims))
    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def _update_similarity(ds_pk, other_ds_pks, sims):
    ds_1 = models.Dataset.objects.get(pk=ds_pk)
    for pk, sim in zip(other_ds_pks, sims):
        ds_2 = models.Dataset.objects.get(pk=pk)
        if ds_1 != ds_2:
            if sim:
                try:
                    models.Similarity.objects.update_or_create(
                        experiment_1=ds_1.experiment,
                        experiment_2=ds_2.experiment,
                        dataset_1=ds_1,
                        dataset_2=ds_2,
                        sim_type='primary',
                    )
                except ValidationError:
                    pass

                try:
                    models.Similarity.objects.update_or_create(
                        experiment_1=ds_2.experiment,
                        experiment_2=ds_1.experiment,
                        dataset_1=ds_2,
                        dataset_2=ds_1,
                        sim_type='primary',
                    )
                except ValidationError:
                    pass
            else:
                models.Similarity.objects.filter(
                    experiment_1=ds_1.experiment,
                    experiment_2=ds_2.experiment,
                    dataset_1=ds_1,
                    dataset_2=ds_2,
                    sim_type='primary',
                ).delete()
                models.Similarity.objects.filter(
                    experiment_1=ds_2.experiment,
                    experiment_2=ds_1.experiment,
                    dataset_1=ds_2,
                    dataset_2=ds_1,
                    sim_type='primary',
                ).delete()
