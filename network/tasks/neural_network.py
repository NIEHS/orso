import random

from celery import group
from celery.decorators import task
from keras import layers
from keras.models import Sequential
from keras.optimizers import SGD
from keras.constraints import max_norm
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from network import models


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

    if len(label_binarizer.classes_) <= 2:
        model = _get_model(len(x[0]), 1, loss='binary_crossentropy')
    else:
        model = _get_model(len(x[0]), len(label_binarizer.classes_))

    history = model.fit(
        scaler.transform(x_training),
        label_binarizer.transform(y_training),
        epochs=20,
        validation_split=0.11,
        verbose=0,
    )
    accuracy = model.evaluate(
        scaler.transform(x_test),
        label_binarizer.transform(y_test),
    )[1]

    # Save
    fn = network.get_nn_model_path()
    model.save(fn)

    network.neural_network_scaler = scaler
    network.neural_network_label_binarizer = label_binarizer
    network.accuracy = accuracy
    network.training_history = history.history
    network.save()


def _get_model(input_dims, output_dims, loss='categorical_crossentropy'):

    model = Sequential()

    model.add(layers.Dropout(0.2, input_shape=(input_dims,)))
    model.add(layers.Dense(units=500, activation='relu',
              kernel_constraint=max_norm(3)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=500, activation='relu',
              kernel_constraint=max_norm(3)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=output_dims, activation='softmax'))

    sgd = SGD(lr=0.0001)

    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])

    return model
