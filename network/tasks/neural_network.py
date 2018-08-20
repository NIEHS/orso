import json
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from celery import group
from celery.decorators import task
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db.models import Q
from keras import backend as K
from keras import layers
from keras.callbacks import EarlyStopping
from keras.constraints import max_norm
from keras.models import load_model, Sequential
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler

from network.tasks.analysis.string_db import (
    ASSEMBLY_TO_ORGANISM, get_organism_to_interaction_partners_dict)
from network import models
from network.tasks.metadata_recommendations import \
    RELEVANT_CELL_TYPE_CATEGORIES, RELEVANT_EPI_CATEGORIES, \
    RELEVANT_GO_CATEGORIES
from network.tasks.recommendations import update_recommendations
from network.tasks.update_pca import generate_selected_loci_df

RELEVANT_TARGET_CATEGORIES = \
    RELEVANT_GO_CATEGORIES + RELEVANT_EPI_CATEGORIES + ['No target']


def create_neural_networks():

    for lg in models.LocusGroup.objects.all():
        for exp_type in models.ExperimentType.objects.all():
            if models.Dataset.objects.filter(
                assembly=lg.assembly,
                experiment__experiment_type=exp_type,
                experiment__project__name='ENCODE',
                experiment__processed=True,
                experiment__revoked=False,
                processed=True,
                revoked=False,
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
                assembly=lg.assembly,
                experiment__experiment_type=exp_type,
                experiment__project__name='ENCODE',
                experiment__processed=True,
                experiment__revoked=False,
                processed=True,
                revoked=False,
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
        experiment__processed=True,
        experiment__revoked=False,
        processed=True,
        revoked=False,
    )

    # Get x and y arrays
    x = []
    y = []

    pca = models.PCA.objects.get(locus_group=network.locus_group,
                                 experiment_type=network.experiment_type)
    df = generate_selected_loci_df(pca, datasets)

    for ds in datasets:
        x.append(df[ds.pk])
        y.append(getattr(ds.experiment, network.metadata_field))

    if network.metadata_field == 'cell_type':
        y = cell_types_to_one_hot(y)
    elif network.metadata_field == 'target':
        y = targets_to_one_hot(y)

    # If one hot vectors contain no categories, remove
    _x = []
    _y = []
    for vec, one_hot in zip(x, y):
        if sum(one_hot) > 0:
            _x.append(vec)
            _y.append(one_hot)
    x = _x
    y = _y

    if not x and not y:
        print('After filtering \'no category\' data, '
              'no datasets remain: {}'.format(':'.join([
                  network.locus_group.assembly.name,
                  network.experiment_type.name,
                  network.locus_group.group_type,
                  network.metadata_field,
              ])))
        return

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

    input_dims = len(x[0])

    if network.metadata_field == 'cell_type':
        output_dims = \
            len(RELEVANT_CELL_TYPE_CATEGORIES)
    elif network.metadata_field == 'target':
        output_dims = \
            len(RELEVANT_TARGET_CATEGORIES)

    unit_count = min(2000, int((input_dims + output_dims) / 2))
    model = _get_model(input_dims, output_dims, unit_count)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.01,
        verbose=0,
        mode='auto',
    )

    history = model.fit(
        scaler.transform(x_training),
        np.array(y_training),
        epochs=100,
        validation_split=0.11,
        verbose=0,
        callbacks=[early_stopping],
    )
    loss, accuracy = model.evaluate(
        scaler.transform(x_test),
        np.array(y_test),
    )

    # Save
    fn = network.get_nn_model_path()
    model.save(fn)
    network.neural_network_file = fn

    network.neural_network_scaler = scaler
    network.loss = loss
    network.accuracy = accuracy
    network.training_history = history.history
    network.save()

    if K.backend() == 'tensorflow':
        K.clear_session()


def _get_model(input_dims, output_dims, unit_count):

    model = Sequential()

    model.add(layers.Dropout(0.5, input_shape=(input_dims,)))
    model.add(layers.Dense(units=unit_count, activation='relu',
              kernel_constraint=max_norm(3)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=unit_count, activation='relu',
              kernel_constraint=max_norm(3)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=output_dims, activation='sigmoid'))

    sgd = SGD(lr=0.1)

    model.compile(loss='binary_crossentropy', optimizer=sgd,
                  metrics=['categorical_accuracy'])

    return model


@task
def generate_training_plots(nn_pk):
    nn = models.NeuralNetwork.objects.get(pk=nn_pk)

    generate_accuracy_training_plot(nn)
    generate_loss_training_plot(nn)


def generate_accuracy_training_plot(neural_network):

    epochs = list(range(len(neural_network.training_history['acc'])))

    plt.plot(epochs, neural_network.training_history['acc'],
             label='Training accuracy')
    plt.plot(epochs, neural_network.training_history['val_acc'],
             label='Validation accuracy')
    plt.axhline(y=neural_network.accuracy, color='black', linestyle='--',
                label='Test accuracy')

    plt.title('Accuracy over training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')

    plt.savefig(neural_network.get_accuracy_plot_path())
    plt.close()


def generate_loss_training_plot(neural_network):

    epochs = list(range(len(neural_network.training_history['loss'])))

    plt.plot(epochs, neural_network.training_history['loss'],
             label='Training loss')
    plt.plot(epochs, neural_network.training_history['val_loss'],
             label='Validation loss')
    plt.axhline(y=neural_network.loss, color='black', linestyle='--',
                label='Test loss')

    plt.title('Loss over training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.savefig(neural_network.get_loss_plot_path())
    plt.close()


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

    dataset.predicted_cell_type_json = json.dumps(
        predict_dataset_field(dataset, 'cell_type'))
    dataset.predicted_target_json = json.dumps(
        predict_dataset_field(dataset, 'target'))

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
        ).get_filtered_vector()

    except (
        IndexError,
        AttributeError,
        models.DatasetIntersectionJson.DoesNotExist,
    ):
        return None

    else:

        if any([
            nn.neural_network_file is None,
            nn.neural_network_scaler is None,
        ]):
            return None

        scaled = nn.neural_network_scaler.transform([intersection_vector])
        model = load_model(nn.neural_network_file.path)

        predictions = model.predict(scaled)[0]

        predicted_classes = []
        if metadata_field == 'cell_type':
            for pred, _class in zip(
                    predictions, RELEVANT_CELL_TYPE_CATEGORIES):
                predicted_classes.append((_class, float(pred)))
        elif metadata_field == 'target':
            for pred, _class in zip(
                    predictions, RELEVANT_TARGET_CATEGORIES):
                predicted_classes.append((_class, float(pred)))

        if K.backend() == 'tensorflow':
            K.clear_session()

        return predicted_classes


def get_predicted_classes(dataset, predicted_field, threshold=0.5):
    class_set = set()

    if predicted_field == 'cell_type':
        prediction_json = dataset.predicted_cell_type_json
    elif predicted_field == 'target':
        prediction_json = dataset.predicted_target_json
    else:
        raise ValueError('Improper predicted_field value.')

    try:
        predictions = json.loads(prediction_json)
        for predicted_class, prediction_value in predictions:
            if prediction_value > threshold:
                class_set.add(predicted_class)
    except TypeError:
        pass

    return class_set


def generate_predicted_sims_df(datasets, identity_only=False):

    relevant_epi_set = set(RELEVANT_EPI_CATEGORIES)
    relevant_go_set = set(RELEVANT_GO_CATEGORIES)

    # Get STRING interaction partners
    gene_dict = defaultdict(set)
    for ds in datasets:
        gene_dict[ds.assembly.name].add(ds.experiment.target)
    interaction_partners = get_organism_to_interaction_partners_dict(gene_dict)

    datasets = list(datasets)

    cell_type_list = \
        [get_predicted_classes(ds, 'cell_type') for ds in datasets]
    target_list = \
        [get_predicted_classes(ds, 'target') for ds in datasets]

    d = {}
    for ds_1, cell_type_set_1, target_set_1 in zip(
            datasets, cell_type_list, target_list):

        comp_values = []

        for ds_2, cell_type_set_2, target_set_2 in zip(
                datasets, cell_type_list, target_list):

            if ds_1 == ds_2:
                comp_values.append(True)
            else:

                is_tf = bool(target_set_1 & target_set_2 & relevant_go_set)
                is_hist = bool(target_set_1 & target_set_2 & relevant_epi_set)
                is_none = bool(target_set_1 & target_set_2 &
                               set(['No target']))

                organism = ASSEMBLY_TO_ORGANISM[ds_1.assembly.name]
                _interaction_partners = interaction_partners[organism]
                target_1 = ds_1.experiment.target
                target_2 = ds_2.experiment.target
                interacting = any([
                    target_1 in _interaction_partners[target_2],
                    target_2 in _interaction_partners[target_1],
                ])

                cell_type_sim = bool(cell_type_set_1 & cell_type_set_2)
                target_sim = any([
                    (is_tf and interacting),
                    is_hist,
                    is_none,
                ])

                comp_values.append(all([
                    ds_1.assembly == ds_2.assembly,
                    ds_1.experiment.experiment_type ==
                    ds_2.experiment.experiment_type,
                    cell_type_sim,
                    target_sim,
                ]))

        series = pd.Series(
            comp_values, index=[ds.pk for ds in datasets])
        d.update({ds_1.pk: series})

    return pd.DataFrame(d)


def update_primary_data_sims_and_recs(experiment_pk):
    update_primary_data_similarities(experiment_pk)
    update_recommendations(experiment_pk, sim_types=['primary'], lock=False)


def update_primary_data_similarities(experiment_pk):
    experiment = models.Experiment.objects.get(pk=experiment_pk)
    datasets = models.Dataset.objects.filter(experiment=experiment)

    # Get relevant datasets
    assemblies = models.Assembly.objects.filter(dataset__in=datasets)
    other_datasets = models.Dataset.objects.filter(
        assembly__in=assemblies,
        experiment__experiment_type=experiment.experiment_type,
    )
    total_datasets = set(datasets) | set(other_datasets)

    sims_df = generate_predicted_sims_df(total_datasets)

    for ds_1 in datasets:
        for ds_2 in other_datasets:
            if all([
                ds_1.experiment != ds_2.experiment,
                ds_1.experiment.experiment_type ==
                ds_2.experiment.experiment_type,
                ds_1.assembly == ds_2.assembly,
            ]):
                if sims_df[ds_1.pk][ds_2.pk]:
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


def update_all_similarities_and_recommendations():
    datasets = models.Dataset.objects.all()
    update_bulk_similarities([ds.pk for ds in datasets])

    user_experiments = models.Experiment.objects.filter(
        Q(owners=True) | Q(favorite__user=True))

    tasks = []
    for experiment in user_experiments:
        tasks.append(update_recommendations.si(
            experiment.pk, sim_types=['primary']))

    job = group(tasks)
    results = job.apply_async()
    results.join()


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


def cell_types_to_one_hot(cell_types):

    cell_type_categories = []

    # Get BRENDA ontology object
    brenda_ont = (models.Ontology.objects.get(name='brenda_tissue_ontology')
                                         .get_ontology_object())
    relevant_cell_type_set = set(RELEVANT_CELL_TYPE_CATEGORIES)

    # Get ENCODE to BRENDA dict
    encode_to_brenda_path = os.path.join(
        settings.ONTOLOGY_DIR, 'encode_to_brenda.json')
    try:
        with open(encode_to_brenda_path) as f:
            encode_cell_type_to_brenda_name = json.load(f)
    except FileNotFoundError:
        print('ENCODE to BRENDA file not found.')
        encode_cell_type_to_brenda_name = dict()

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
            cell_type_categories.append(
                set([brenda_ont.term_to_name[term] for term in parent_set]) &
                relevant_cell_type_set
            )
        else:
            cell_type_categories.append(set())

    _categories = []
    for x in cell_type_categories:
        _categories.append([
            int(cat in x) for cat in RELEVANT_CELL_TYPE_CATEGORIES])
    cell_type_categories = _categories

    return cell_type_categories


def targets_to_one_hot(targets):

    target_categories = []

    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()
    epi_ont = models.Ontology.objects.get(
        name='epigenetic_modification_ontology').get_ontology_object()
    relevant_go_set = set(RELEVANT_GO_CATEGORIES)
    relevant_epi_set = set(RELEVANT_EPI_CATEGORIES)

    for target in targets:

        categories = set()

        if target:

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

        else:

            categories.add('No target')

        target_categories.append(categories)

    _categories = []
    for x in target_categories:
        _categories.append([
            int(cat in x) for cat in RELEVANT_TARGET_CATEGORIES])
    target_categories = _categories

    return target_categories
