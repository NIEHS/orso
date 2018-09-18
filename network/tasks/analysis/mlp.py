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
from keras import backend as K
from keras import layers
from keras.callbacks import EarlyStopping
from keras.constraints import max_norm
from keras.models import load_model, Sequential
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler

from network import models
from network.tasks.analysis.pca import generate_selected_loci_df
from network.tasks.similarity import \
    get_cell_type_classes_list, get_target_classes_list, \
    RELEVANT_CELL_TYPE_CATEGORIES, RELEVANT_EPI_CATEGORIES, \
    RELEVANT_GO_CATEGORIES

RELEVANT_TARGET_CATEGORIES = \
    RELEVANT_GO_CATEGORIES + RELEVANT_EPI_CATEGORIES + ['No target']


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
            ).count() >= 20:

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
        y = get_cell_type_one_hot_list(y)
    elif network.metadata_field == 'target':
        y = get_target_one_hot_list(y)

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

    return dataset


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

        if metadata_field == 'cell_type':
            relevant_classes = RELEVANT_CELL_TYPE_CATEGORIES
        elif metadata_field == 'target':
            relevant_classes = RELEVANT_TARGET_CATEGORIES

        predicted_classes = []
        for pred, _class in zip(
                predictions, relevant_classes):
            predicted_classes.append((_class, float(pred)))

        if K.backend() == 'tensorflow':
            K.clear_session()

        return predicted_classes


def get_one_hot_list(field_list, to_classes_function, relevant_class_list):
    one_hot_list = []
    for classes in to_classes_function(field_list):
        one_hot_list.append(to_one_hot(classes, relevant_class_list))
    return one_hot_list


def to_one_hot(classes, relevant_class_list):
    return [int(cls in classes) for cls in relevant_class_list]


def get_cell_type_one_hot_list(cell_types):
    return get_one_hot_list(
        cell_types,
        get_cell_type_classes_list,
        RELEVANT_CELL_TYPE_CATEGORIES,
    )


def get_target_one_hot_list(targets):
    return get_one_hot_list(
        targets,
        get_target_classes_list,
        RELEVANT_TARGET_CATEGORIES,
    )
