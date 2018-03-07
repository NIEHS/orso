import json
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy
from celery.decorators import task
from django.conf import settings
from django.db.models import Q
from matplotlib.colors import rgb2hex
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from analysis.utils import (
    generate_intersection_df, generate_pca_transformed_df)
from network import models

TOTAL_HISTONE_MARKS = [
    'H2AFZ',
    'H2AK5ac',
    'H2AK9ac',
    'H2BK120ac',
    'H2BK12ac',
    'H2BK15ac',
    'H2BK20ac',
    'H2BK5ac',
    'H3F3A',
    'H3K14ac',
    'H3K18ac',
    'H3K23ac',
    'H3K23me2',
    'H3K27ac',
    'H3K27me3',
    'H3K36me3',
    'H3K4ac',
    'H3K4me1',
    'H3K4me2',
    'H3K4me3',
    'H3K56ac',
    'H3K79me1',
    'H3K79me2',
    'H3K79me3',
    'H3K9ac',
    'H3K9me1',
    'H3K9me2',
    'H3K9me3',
    'H3T11ph',
    'H3ac',
    'H4K12ac',
    'H4K20me1',
    'H4K5ac',
    'H4K8ac',
    'H4K91ac',
]
HIGHLY_REPRESENTED_MARKS = [
    'H2AFZ',
    'H3K27ac',
    'H3K27me3',
    'H3K36me3',
    'H3K4me1',
    'H3K4me2',
    'H3K4me3',
    'H3K79me2',
    'H3K9ac',
    'H3K9me3',
]
TARGET_VECTORS = HIGHLY_REPRESENTED_MARKS


# Due to compatiblity issues between Celery and scikit-learn, update_pca
# cannot be managed by Celery.
def add_or_update_pcas(**kwargs):
    '''
    Perform PCA analysis observing the datasets.
    '''
    pcas = []
    for lg in models.LocusGroup.objects.all():
        for exp_type in models.ExperimentType.objects.all():
            if models.Dataset.objects.filter(
                assembly=lg.assembly,
                experiment__experiment_type=exp_type,
            ).count() >= 3:  # Verify that there are at least 3 datasets
                pcas.append(models.PCA.objects.get_or_create(
                    locus_group=lg,
                    experiment_type=exp_type,
                )[0])

    for pca in pcas:
        update_pca(pca, **kwargs)


def update_pca(pca, **kwargs):
    print(pca.pk)

    fit_pca(pca, **kwargs)
    transform_pca_datasets(pca.pk)
    set_pca_plot(pca)
    fit_nn(pca)
    predict_nn(pca.pk)


# Incompatible with Celery
def fit_pca(pca, size_threshold=200, threads=1):
    datasets = models.Dataset.objects.filter(
        assembly=pca.locus_group.assembly,
        experiment__experiment_type=pca.experiment_type,
        experiment__project__name='ENCODE',
    )

    df = generate_intersection_df(pca.locus_group, pca.experiment_type,
                                  datasets=datasets)

    if df.shape[1] >= 3:
        # Initial filtering
        if pca.locus_group.group_type in ['promoter', 'genebody', 'mRNA']:

            query = Q(gene__annotation__assembly=pca.locus_group.assembly)
            # Filter for selected transcripts (highest expression per gene)
            query &= Q(selecting__isnull=False)
            # Filter by prefixes implying limited curation ('LOC', etc.)
            for prefix in ['LINC', 'LOC']:
                query &= ~Q(gene__name__startswith=prefix)
            # Filter by suffixes implying antisense transcripts
            for suffix in ['-AS', '-AS1', '-AS2', '-AS3', '-AS4', '-AS5']:
                query &= ~Q(gene__name__endswith=suffix)

            selected_transcripts = models.Transcript.objects.filter(query)

            selected_locus_pks = models.Locus.objects.filter(
                transcript__in=selected_transcripts,
                group=pca.locus_group,
            ).values_list('pk', flat=True)

            df = df.loc[list(selected_locus_pks)]

            # Filter out shorter transcripts
            if pca.experiment_type.name != 'microRNA-seq':
                selected_transcripts = [
                    t for t in selected_transcripts
                    if t.end - t.start + 1 >= size_threshold
                ]
                selected_locus_pks = models.Locus.objects.filter(
                    transcript__in=selected_transcripts,
                    group=pca.locus_group,
                ).values_list('pk', flat=True)
                df = df.loc[list(selected_locus_pks)]

            # Select 2nd and 3rd quartiles, in terms of variance and signal
            med_iqr = (
                df.median(axis=1).quantile(q=0.25),
                df.median(axis=1).quantile(q=0.75),
            )
            var_iqr = (
                df.var(axis=1).quantile(q=0.25),
                df.var(axis=1).quantile(q=0.75),
            )
            df = df.loc[
                (df.median(axis=1) >= med_iqr[0]) &
                (df.median(axis=1) <= med_iqr[1]) &
                (df.var(axis=1) >= var_iqr[0]) &
                (df.var(axis=1) <= var_iqr[1])
            ]

        elif pca.locus_group.group_type in ['enhancer']:
            pass

        pca_model = PCA(n_components=3)
        rf = RandomForestClassifier(n_estimators=1000, n_jobs=threads)

        # Get cell lines and targets
        _order = df.columns.values.tolist()
        _datasets = list(models.Dataset.objects.filter(pk__in=_order))
        _datasets.sort(key=lambda ds: _order.index(ds.pk))

        cell_types = [ds.experiment.cell_type for ds in _datasets]
        targets = [ds.experiment.target for ds in _datasets]

        # Apply to RF classifier, get importances
        _data = numpy.transpose(numpy.array(df))
        _loci = list(df.index)

        cell_type_importances = rf.fit(
            _data, cell_types).feature_importances_
        target_importances = rf.fit(
            _data, targets).feature_importances_
        totals = [x + y for x, y in zip(cell_type_importances,
                                        target_importances)]

        # Filter by importances
        filtered_loci = \
            [locus for locus, total in sorted(zip(_loci, totals),
                                              key=lambda x: -x[1])][:1000]
        df = df.loc[filtered_loci]

        # Filter datasets by Mahalanobis distance after PCA
        df_filtered = df

        if df.shape[1] >= 10:
            _datasets = df.columns.values.tolist()
            _data = numpy.transpose(numpy.array(df))

            fitted = pca_model.fit_transform(_data)

            mean = numpy.mean(fitted, axis=0)
            cov = numpy.cov(fitted, rowvar=False)
            inv = numpy.linalg.inv(cov)
            m_dist = []
            for vector in fitted:
                m_dist.append(mahalanobis(vector, mean, inv))

            Q1 = numpy.percentile(m_dist, 25)
            Q3 = numpy.percentile(m_dist, 75)
            cutoff = Q3 + 1.5 * (Q3 - Q1)

            selected_datasets = []
            for dist, ds in zip(m_dist, _datasets):
                if dist < cutoff:
                    selected_datasets.append(ds)

            if selected_datasets:
                df_filtered = df[selected_datasets]

        # Fit with filtered data
        _data = numpy.transpose(numpy.array(df_filtered))

        fitted = pca_model.fit_transform(_data)
        if fitted.shape[0] > 1:
            cov = numpy.cov(fitted, rowvar=False)
            if fitted.shape[0] > 3:
                inv = numpy.linalg.inv(cov)
            else:
                inv = numpy.linalg.pinv(cov)
        else:
            cov = None
            inv = None

        # Fit values to PCA, create the associated PCA plot
        _data = numpy.transpose(numpy.array(df))
        _datasets = df.columns.values.tolist()

        transformed = pca_model.transform(_data)
        pca_plot = []
        for pk, transformed_data in zip(_datasets, transformed):
            ds = models.Dataset.objects.get(pk=pk)
            pca_plot.append({
                'dataset_pk': ds.pk,
                'experiment_pk': ds.experiment.pk,
                'experiment_cell_type': ds.experiment.cell_type,
                'experiment_target': ds.experiment.target,
                'transformed_values': transformed_data.tolist(),
            })

        # Update or create the PCA object
        pca.plot = pca_plot
        pca.pca = pca_model
        pca.covariation_matrix = cov
        pca.inverse_covariation_matrix = inv
        pca.save()

        pca.selected_loci.clear()

        _order = list(df.index)
        _loci = list(models.Locus.objects.filter(pk__in=_order))
        _loci.sort(key=lambda locus: _order.index(locus.pk))

        _pca_locus_orders = []
        for i, locus in enumerate(_loci):
            _pca_locus_orders.append(models.PCALocusOrder(
                pca=pca,
                locus=locus,
                order=i,
            ))
        models.PCALocusOrder.objects.bulk_create(
            _pca_locus_orders)


# Incompatible with Celery
def fit_nn(pca, sample_num=100000):

    df = generate_pca_transformed_df(pca)
    metadata_sims = models.DatasetMetadataDistance.objects.filter(
        dataset_1__pca=pca,
        dataset_2__pca=pca,
    )

    data_vector_list = []
    is_similar_list = []

    for sim in metadata_sims:
        vec_1 = list(df[sim.dataset_1.pk])
        vec_2 = list(df[sim.dataset_2.pk])

        data_vector_list.append(vec_1 + vec_2)
        is_similar_list.append(int(sim.distance))

        data_vector_list.append(vec_2 + vec_1)
        is_similar_list.append(int(sim.distance))

    if sample_num > len(data_vector_list):
        sample_num = len(data_vector_list)

    data_vector_training = []
    is_similar_training = []
    for x, y in random.sample(list(zip(
            data_vector_list, is_similar_list)), sample_num):
        data_vector_training.append(x)
        is_similar_training.append(y)

    scaler = StandardScaler()
    scaler.fit(data_vector_training)

    clf = MLPClassifier(
        solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(
        scaler.transform(data_vector_training),
        is_similar_training,
    )

    pca.neural_network = clf
    pca.neural_network_scaler = scaler
    pca.save()


@task
def transform_pca_datasets(pca_pk):
    pca = models.PCA.objects.get(pk=pca_pk)

    order = models.PCALocusOrder.objects.filter(pca=pca).order_by('order')
    loci = [x.locus for x in order]

    for dij in models.DatasetIntersectionJson.objects.filter(
        locus_group=pca.locus_group,
        dataset__experiment__experiment_type=pca.experiment_type,
    ):

        intersection_values = json.loads(dij.intersection_values)

        locus_values = dict()
        for val, pk in zip(
            intersection_values['normalized_values'],
            intersection_values['locus_pks']
        ):
            locus_values[pk] = val

        normalized_values = []
        for locus in loci:
            try:
                normalized_values.append(locus_values[locus.pk])
            except IndexError:
                normalized_values.append(0)

        transformed_values = pca.pca.transform([normalized_values])[0]
        models.PCATransformedValues.objects.update_or_create(
            pca=pca,
            dataset=dij.dataset,
            defaults={
                'transformed_values': transformed_values.tolist(),
            },
        )


@task
def predict_nn(pca_pk):
    pca = models.PCA.objects.get(pk=pca_pk)

    datasets = models.Dataset.objects.filter(
        assembly=pca.locus_group.assembly,
        experiment__experiment_type=pca.experiment_type,
    ).order_by('pk')

    for i, ds_1 in enumerate(datasets):
        for j, ds_2 in enumerate(datasets):

            if i < j:

                # COPIED FROM PROCESS_DATASETS.PY
                try:
                    vec_1 = models.PCATransformedValues.objects.get(
                        dataset=ds_1, pca=pca).transformed_values
                except models.PCATransformedValues.DoesNotExist:
                    print('Transformed values do not exist for {}.'.format(
                        ds_1.name))
                    vec_1 = None

                try:
                    vec_2 = models.PCATransformedValues.objects.get(
                        dataset=ds_2, pca=pca).transformed_values
                except models.PCATransformedValues.DoesNotExist:
                    print('Transformed values do not exist for {}.'.format(
                        ds_2.name))
                    vec_2 = None

                if vec_1 and vec_2:
                    vec = pca.neural_network_scaler.transform([vec_1 + vec_2])
                    sim = pca.neural_network.predict(vec)[0]

                    if ds_1.pk < ds_2.pk:
                        _ds_1, _ds_2 = ds_1, ds_2
                    else:
                        _ds_1, _ds_2 = ds_2, ds_1
                    models.DatasetDataDistance.objects.update_or_create(
                        dataset_1=_ds_1,
                        dataset_2=_ds_2,
                        defaults={
                            'distance': sim,
                        },
                    )
                # COPIED FROM PROCESS_DATASETS.PY


def set_pca_plot(pca):

    def get_plot(pca):
        plot = {
            'color_choices': [
                'Cell type',
                'Target',
            ],
            'points': {
                'Histone': [],
                'Control': [],
                'Other': [],
            },
            'vectors': {},
        }

        datasets = \
            list(models.Dataset.objects.filter(pcatransformedvalues__pca=pca))
        transformed_values = []
        for ds in datasets:
            transformed_values.append(ds.pcatransformedvalues_set
                                      .get(pca=pca).transformed_values)

        cmap = plt.get_cmap('hsv')

        target_color_path = os.path.join(
            settings.COLOR_KEY_DIR, 'target.json')
        if os.path.exists(target_color_path):
            with open(target_color_path) as f:
                target_to_color = json.load(f)
        else:
            target_set = set([ds.experiment.target for ds in datasets])
            target_to_color = dict()
            for i, target in enumerate(list(target_set)):
                j = i % 20
                index = ((j % 2) * (50) + (j // 2) * (5)) / 100
                target_to_color[target] = rgb2hex(cmap(index))

        cell_type_color_path = os.path.join(
            settings.COLOR_KEY_DIR, 'cell_type.json')
        if os.path.exists(cell_type_color_path):
            with open(cell_type_color_path) as f:
                cell_type_to_color = json.load(f)
        else:
            cell_type_set = set([ds.experiment.cell_type for ds in datasets])
            cell_type_to_color = dict()
            for i, cell_type in enumerate(list(cell_type_set)):
                j = i % 20
                index = ((j % 2) * (50) + (j // 2) * (5)) / 100
                cell_type_to_color[cell_type] = rgb2hex(cmap(index))

        vector_categories = defaultdict(list)
        for ds, values in zip(datasets, transformed_values):
            target = ds.experiment.target
            if target in TARGET_VECTORS:
                vector_categories[target].append(values)
        for vector, values in vector_categories.items():
            if vector in target_to_color:
                color = target_to_color[vector]
            else:
                color = rgb2hex(cmap(0))
            plot['vectors'].update(
                {
                    vector: {
                        'point': numpy.mean(values, axis=0).tolist(),
                        'color': color,
                        'label': vector,
                    }
                })

        for ds, values in zip(datasets, transformed_values):
            colors = dict()
            colors.update({'None': '#A9A9A9'})

            target = ds.experiment.target
            if target in target_to_color:
                colors.update({'Target': target_to_color[target]})
            else:
                print('Target not found: \"{}\"'.format(target))
                colors.update({'Target': '#A9A9A9'})

            cell_type = ds.experiment.cell_type
            if cell_type in cell_type_to_color:
                colors.update({'Cell type': cell_type_to_color[cell_type]})
            else:
                print('Cell type not found: \"{}\"'.format(cell_type))
                colors.update({'Cell type': '#A9A9A9'})

            point = {
                'experiment_name': ds.experiment.name,
                'dataset_name': ds.name,
                'experiment_pk': ds.experiment.pk,
                'dataset_pk': ds.pk,
                'experiment_cell_type': ds.experiment.cell_type,
                'experiment_target': ds.experiment.target,
                'transformed_values': values,
                'colors': colors,
            }

            if target in ['Control']:
                plot['points']['Control'].append(point)
            elif target in TOTAL_HISTONE_MARKS:
                plot['points']['Histone'].append(point)
            else:
                plot['points']['Other'].append(point)

        return plot

    def get_explained_variance(pca):
        return pca.pca.explained_variance_ratio_.tolist()

    def get_components(pca):
        components = []

        if pca.locus_group.group_type in ['genebody', 'promoter', 'mRNA']:
            locus_names = [
                x.locus.transcript.gene.name for x in
                models.PCALocusOrder.objects.filter(pca=pca).order_by('order')
            ]
        elif pca.locus_group.group_type in ['enhancer']:
            locus_names = [
                x.locus.enhancer.name for x in
                models.PCALocusOrder.objects.filter(pca=pca).order_by('order')
            ]

        for _component in pca.pca.components_:
            components.append(
                sorted(
                    zip(locus_names, _component),
                    key=lambda x: -abs(x[1]),
                )[:20]
            )

        return components

    pca.plot = json.dumps({
        'plot': get_plot(pca),
        'explained_variance': get_explained_variance(pca),
        'components': get_components(pca),
    })
    pca.save()
