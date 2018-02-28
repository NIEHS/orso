import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy
from celery import group
from celery.decorators import task
from django.conf import settings
from django.db.models import Q
from matplotlib.colors import rgb2hex
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from analysis.utils import generate_intersection_df
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


@task
def add_or_update_pca(datasets, **kwargs):
    '''
    Perform PCA analysis observing the datasets.
    '''
    dataset_pks = set([ds.pk for ds in datasets])

    for lg in sorted(models.LocusGroup.objects.all(), key=lambda x: x.pk):
        for exp_type in sorted(models.ExperimentType.objects.all(),
                               key=lambda x: x.pk):

            subset = set(models.Dataset.objects.filter(
                assembly=lg.assembly,
                experiment__experiment_type=exp_type,
            ).values_list('pk', flat=True))

            _pca_analysis_json(
                lg.pk, exp_type.pk, list(dataset_pks & subset), **kwargs)


@task
def _pca_analysis_json(locusgroup_pk, experimenttype_pk, dataset_pks,
                       size_threshold=200, threads=1):
    locus_group = models.LocusGroup.objects.get(pk=locusgroup_pk)
    experiment_type = models.ExperimentType.objects.get(pk=experimenttype_pk)
    datasets = models.Dataset.objects.filter(pk__in=list(dataset_pks))

    df = generate_intersection_df(locus_group, experiment_type,
                                  datasets=datasets)

    if df.shape[1] >= 3:

        # Initial filtering
        if locus_group.group_type in ['promoter', 'genebody', 'mRNA']:

            query = Q(gene__annotation__assembly=locus_group.assembly)
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
                group=locus_group,
            ).values_list('pk', flat=True)

            df = df.loc[list(selected_locus_pks)]

            # Filter out shorter transcripts
            if experiment_type.name != 'microRNA-seq':
                selected_transcripts = [
                    t for t in selected_transcripts
                    if t.end - t.start + 1 >= size_threshold
                ]
                selected_locus_pks = models.Locus.objects.filter(
                    transcript__in=selected_transcripts,
                    group=locus_group,
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

        elif locus_group.group_type in ['enhancer']:
            pass

        pca = PCA(n_components=3)
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

            fitted = pca.fit_transform(_data)

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

        fitted = pca.fit_transform(_data)
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

        transformed = pca.transform(_data)
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
        pca_object, created = models.PCA.objects.update_or_create(
            locus_group=locus_group,
            experiment_type=experiment_type,
            defaults={
                'plot': pca_plot,
                'pca': pca,
                'covariation_matrix': cov,
                'inverse_covariation_matrix': inv,
            },
        )

        # Set the PCA to loci relationships
        if not created:
            pca_object.selected_loci.clear()

        _order = list(df.index)
        _loci = list(models.Locus.objects.filter(pk__in=_order))
        _loci.sort(key=lambda locus: _order.index(locus.pk))

        _pca_locus_orders = []
        for i, locus in enumerate(_loci):
            _pca_locus_orders.append(models.PCALocusOrder(
                pca=pca_object,
                locus=locus,
                order=i,
            ))
        models.PCALocusOrder.objects.bulk_create(
            _pca_locus_orders)


@task
def _pca_analysis(locusgroup_pk, experimenttype_pk, dataset_pks,
                  size_threshold=200):

    locus_group = models.LocusGroup.objects.get(pk=locusgroup_pk)
    experiment_type = models.ExperimentType.objects.get(pk=experimenttype_pk)

    if locus_group.group_type in ['promoter', 'genebody', 'mRNA']:

        # Get all transcripts associated with the locus group and that are the
        # selected transcript for a gene
        transcripts = models.Transcript.objects.filter(
            gene__annotation__assembly=locus_group.assembly,
            selecting__isnull=False,
        )

        # Filter transcripts by size if not microRNA-seq
        if experiment_type.name != 'microRNA-seq':
            transcripts = [
                t for t in transcripts
                if t.end - t.start + 1 >= size_threshold
            ]

        # Get loci associated with the transcripts and locus group
        loci = models.Locus.objects.filter(
            group=locus_group, transcript__in=transcripts)

    elif locus_group.group_type in ['enhancer']:
        loci = models.Locus.objects.filter(group=locus_group)

    datasets = models.Dataset.objects.filter(pk__in=list(dataset_pks))

    loci_num = len(models.Locus.objects.filter(group=locus_group))
    temp_datasets = []
    for ds in datasets:
        intersection_num = len(models.DatasetIntersection.objects.filter(
            dataset=ds,
            locus__group=locus_group,
        ))
        if loci_num == intersection_num:
            temp_datasets.append(ds)
    datasets = temp_datasets

    if len(datasets) >= 3:
        pca = PCA(n_components=3)
        rf = RandomForestClassifier(n_estimators=1000)

        intersection_values = dict()
        experiment_pks = dict()
        cell_types = dict()
        targets = dict()

        # Get associated data
        for ds in datasets:
            exp = models.Experiment.objects.get(dataset=ds)
            experiment_pks[ds] = exp.pk
            cell_types[ds] = exp.cell_type
            targets[ds] = exp.target

            intersection_values[ds] = dict()

            loci = sorted(loci, key=lambda x: x.pk)
            intersections = (
                models.DatasetIntersection
                      .objects.filter(dataset=ds,
                                      locus__in=loci)
                      .order_by('locus__pk')
            )

            for locus, intersection in zip(loci, intersections):
                intersection_values[ds][locus] = intersection.normalized_value

        # Filter loci by RF importance
        _intersection_values = []
        _cell_types = []
        _targets = []

        for ds in datasets:
            _intersection_values.append([])
            for locus in loci:
                _intersection_values[-1].append(
                    intersection_values[ds][locus])
            _cell_types.append(cell_types[ds])
            _targets.append(targets[ds])

        cell_type_importances = rf.fit(
            _intersection_values, _cell_types).feature_importances_
        target_importances = rf.fit(
            _intersection_values, _targets).feature_importances_
        totals = [x + y for x, y in zip(cell_type_importances,
                                        target_importances)]
        filtered_loci = \
            [locus for locus, total in sorted(zip(loci, totals),
                                              key=lambda x: -x[1])][:1000]

        # Filter datasets by Mahalanobis distance after PCA
        filtered_datasets = []
        if len(datasets) >= 10:
            _intersection_values = []
            for ds in datasets:
                _intersection_values.append([])
                for locus in filtered_loci:
                    _intersection_values[-1].append(
                        intersection_values[ds][locus])

            fitted = pca.fit_transform(_intersection_values)

            mean = numpy.mean(fitted, axis=0)
            cov = numpy.cov(fitted, rowvar=False)
            inv = numpy.linalg.inv(cov)
            m_dist = []
            for vector in fitted:
                m_dist.append(mahalanobis(vector, mean, inv))

            Q1 = numpy.percentile(m_dist, 25)
            Q3 = numpy.percentile(m_dist, 75)
            cutoff = Q3 + 1.5 * (Q3 - Q1)

            filtered_datasets = []
            for dist, ds in zip(m_dist, datasets):
                if dist < cutoff:
                    filtered_datasets.append(ds)
        if len(filtered_datasets) <= 1:
            filtered_datasets = datasets

        # Fit PCA with filtered transcripts and filtered datasets
        _intersection_values = []
        for ds in filtered_datasets:
            _intersection_values.append([])
            for locus in filtered_loci:
                _intersection_values[-1].append(
                    intersection_values[ds][locus])
        fitted = pca.fit_transform(_intersection_values)
        if len(fitted) > 1:
            cov = numpy.cov(fitted, rowvar=False)
            if len(_intersection_values) > 3:
                inv = numpy.linalg.inv(cov)
            else:
                inv = numpy.linalg.pinv(cov)
        else:
            cov = None
            inv = None

        # Fit values to PCA, create the associated PCA plot
        fitted_values = []
        pca_plot = []
        for ds in datasets:
            _intersection_values = []
            for locus in filtered_loci:
                _intersection_values.append(
                    intersection_values[ds][locus])
            fitted_values.append(pca.transform([_intersection_values])[0])

            pca_plot.append({
                'dataset_pk': ds.pk,
                'experiment_pk': experiment_pks[ds],
                'experiment_cell_type': cell_types[ds],
                'experiment_target': targets[ds],
                'transformed_values': fitted_values[-1].tolist(),
            })

        # Update or create the PCA object
        pca, created = models.PCA.objects.update_or_create(
            locus_group=locus_group,
            experiment_type=experiment_type,
            defaults={
                'plot': pca_plot,
                'pca': pca,
                'covariation_matrix': cov,
                'inverse_covariation_matrix': inv,
            },
        )

        # Set the PCA to loci relationships
        if not created:
            pca.selected_loci.clear()
        _pca_locus_orders = []
        for i, locus in enumerate(filtered_loci):
            _pca_locus_orders.append(models.PCALocusOrder(
                pca=pca,
                locus=locus,
                order=i,
            ))
        models.PCALocusOrder.objects.bulk_create(
            _pca_locus_orders)


@task
def add_or_update_pca_transformed_values():
    tasks = []

    for pca in models.PCA.objects.all():
        for dataset in models.Dataset.objects.filter(
            assembly=pca.locus_group.assembly,
            experiment__experiment_type=pca.experiment_type,
        ):
            tasks.append(_add_or_update_pca_transformed_values.s(
                dataset.pk, pca.pk
            ))

    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def _add_or_update_pca_transformed_values(dataset_pk, pca_pk):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    pca = models.PCA.objects.get(pk=pca_pk)

    order = models.PCALocusOrder.objects.filter(pca=pca).order_by('order')
    loci = [x.locus for x in order]

    intersection_values = []
    missing_values = False
    for locus in loci:
        try:
            intersection_values.append(
                models.DatasetIntersection.objects.get(
                    dataset=dataset, locus=locus).normalized_value
            )
        except models.DatasetIntersection.DoesNotExist:
            print('Missing intersection: Dataset: {}; '
                  'Locus: {}. Break.'.format(str(dataset.pk), str(locus.pk)))
            missing_values = True
            break

    if not missing_values:
        transformed_values = pca.pca.transform([intersection_values])[0]
        models.PCATransformedValues.objects.update_or_create(
            pca=pca,
            dataset=dataset,
            defaults={
                'transformed_values': transformed_values.tolist(),
            },
        )


@task
def add_or_update_pca_transformed_values_json():
    tasks = []

    for pca in models.PCA.objects.all():
        for dij in models.DatasetIntersectionJson.objects.filter(
            locus_group=pca.locus_group,
            dataset__experiment__experiment_type=pca.experiment_type,
        ):
            tasks.append(_add_or_update_pca_transformed_values_json.s(
                dij.pk, pca.pk
            ))

    job = group(tasks)
    results = job.apply_async()
    results.join()


@task
def _add_or_update_pca_transformed_values_json(dij_pk, pca_pk):
    dij = models.DatasetIntersectionJson.objects.get(pk=dij_pk)
    pca = models.PCA.objects.get(pk=pca_pk)

    order = models.PCALocusOrder.objects.filter(pca=pca).order_by('order')
    loci = [x.locus for x in order]

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
