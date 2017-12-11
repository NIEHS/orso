from celery.decorators import task
from celery import group

from . import models

import numpy
from sklearn.metrics import jaccard_similarity_score
from sklearn.decomposition import PCA

from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import RandomForestClassifier

from analysis import score

from progress.bar import Bar


@task
def update_user_based_recommendations():
    experiments = models.Experiment.objects.all().order_by('pk')
    users = models.MyUser.objects.all().order_by('pk')
    user_vectors = dict()

    # Create user vectors
    print('Creating user vectors...')
    bar = Bar('Processing', max=len(users))
    for user in users:
        user_vectors[user] = []
        pk_set = set(models.Experiment.objects.filter(
            experimentfavorite__owner=user).values_list('pk'))
        for exp in experiments:
            if exp.pk in pk_set:
                user_vectors[user].append(1)
            else:
                user_vectors[user].append(0)
        bar.next()
    bar.finish()

    print('Updating user-based recommendations...')
    bar = Bar('Processing', max=len(users))
    for user in users:
        relevant_assemblies = models.Assembly.objects.filter(
            dataset__experiment__owners=user)
        relevant_experiment_types = models.ExperimentType.objects.filter(
            experiment__owners=user)

        relevant_experiments = models.Experiment.objects.filter(
            dataset__assembly__in=relevant_assemblies,
            experiment_type__in=relevant_experiment_types,
        ).exclude(owners=user)

        # Remove any scores that are no longer relevant
        models.UserToExperimentSimilarity.objects.exclude(
            experiment__in=relevant_experiments).delete()

        # Update scores for relevant experiments
        for exp in relevant_experiments:

            favoriting_users = models.MyUser.objects.filter(
                experimentfavorite__favorite=exp)

            sim_score = sum([jaccard_similarity_score(
                user_vectors[user],
                user_vectors[other_user],
            ) for other_user in favoriting_users])

            models.UserToExperimentSimilarity.objects.update_or_create(
                user=user,
                experiment=exp,
                defaults={
                    'score': sim_score,
                }
            )

        bar.next()
    bar.finish()


@task
def add_or_update_pca(datasets):
    '''
    Perform PCA analysis observing the datasets.
    '''
    # tasks = []
    #
    dataset_pks = set([ds.pk for ds in datasets])

    for lg in sorted(models.LocusGroup.objects.all(), key=lambda x: x.pk):
        for exp_type in sorted(models.ExperimentType.objects.all(),
                               key=lambda x: x.pk):

            subset = set(models.Dataset.objects.filter(
                assembly=lg.assembly,
                experiment__experiment_type=exp_type,
            ).values_list('pk', flat=True))

            # tasks.append(_pca_analysis.s(
            #     lg.pk, exp_type.pk, list(dataset_pks & subset)))

            # print(lg.pk, exp_type.pk)
            #
            _pca_analysis(lg.pk, exp_type.pk, list(dataset_pks & subset))

#     job = group(tasks)
#     results = job.apply_async()
#     results.join()


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
            _intersection_values, _cell_types).feature_importances_
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
            print('Missing intersection: Dataset: {}; Locus: {}.'.format(
                str(dataset.pk), str(locus.pk)))
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
def update_dataset_data_scores(datasets, quiet=False):
    '''
    Update or create dataset data distance values.
    '''
    updated = set()

    bar_max = 0
    for ds in datasets:
        bar_max += models.Dataset.objects.filter(
            assembly=ds.assembly,
            experiment__experiment_type=ds.experiment.experiment_type,
        ).count()
    bar = Bar('Processing', max=bar_max)

    for ds_1 in datasets:
        for ds_2 in models.Dataset.objects.filter(
            assembly=ds_1.assembly,
            experiment__experiment_type=ds_1.experiment.experiment_type,
        ):

            _ds_1, _ds_2 = sorted([ds_1, ds_2], key=lambda x: x.pk)
            exp_type_1 = _ds_1.experiment.experiment_type
            exp_type_2 = _ds_2.experiment.experiment_type
            if all([
                (_ds_1, _ds_2) not in updated,
                _ds_1 != _ds_2,
                models.PCATransformedValues.objects.filter(
                    dataset=_ds_1,
                    pca__locus_group__group_type=exp_type_1.relevant_regions,
                ).exists(),
                models.PCATransformedValues.objects.filter(
                    dataset=_ds_2,
                    pca__locus_group__group_type=exp_type_2.relevant_regions,
                ).exists(),
            ]):

                distance = score.score_datasets_by_pca_distance(_ds_1, _ds_2)
                models.DatasetDataDistance.objects.update_or_create(
                    dataset_1=_ds_1,
                    dataset_2=_ds_2,
                    defaults={
                        'distance': distance,
                    },
                )
                updated.add((_ds_1, _ds_2))

            bar.next()

    bar.finish()


@task
def update_dataset_metadata_scores(datasets):
    '''
    Update or create dataset metadata distance values.
    '''
    updated = set()

    bar_max = 0
    for ds in datasets:
        bar_max += models.Dataset.objects.filter(
            assembly=ds.assembly,
            experiment__experiment_type=ds.experiment.experiment_type,
        ).count()
    bar = Bar('Processing', max=bar_max)

    onts = []
    onts.append(models.Ontology.objects.get(name='cell_ontology'))
    onts.append(models.Ontology.objects.get(name='cell_line_ontology'))
    onts.append(models.Ontology.objects.get(name='brenda_tissue_ontology'))
    cell_ont_list = [ont.get_ontology_object() for ont in onts]

    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()

    for ds_1 in datasets:
        for ds_2 in models.Dataset.objects.filter(
            assembly=ds_1.assembly,
            experiment__experiment_type=ds_1.experiment.experiment_type,
        ):

            _ds_1, _ds_2 = sorted([ds_1, ds_2], key=lambda x: x.pk)
            if all([
                (_ds_1, _ds_2) not in updated,
                _ds_1 != _ds_2,
            ]):

                exp_1 = models.Experiment.objects.get(dataset=_ds_1)
                exp_2 = models.Experiment.objects.get(dataset=_ds_2)

                total_sim = 0

                cell_ont_sims = []
                for ont_obj in cell_ont_list:
                    sim = ont_obj.get_word_similarity(
                        exp_1.cell_type, exp_2.cell_type, metric='lin')
                    if sim:
                        cell_ont_sims.append(sim)
                if cell_ont_sims:
                    total_sim += max(cell_ont_sims)

                gene_ont_sim = gene_ont.get_word_similarity(
                    exp_1.target, exp_2.target, metric='jaccard',
                    weighting='information_content')
                if gene_ont_sim:
                    total_sim += gene_ont_sim

                models.DatasetMetadataDistance.objects.update_or_create(
                    dataset_1=_ds_1,
                    dataset_2=_ds_2,
                    defaults={
                        'distance': total_sim,
                    },
                )

            bar.next()

    bar.finish()


@task
def update_experiment_data_scores(experiments):
    '''
    Update or create experiment data distance values.
    '''
    updated = set()

    bar_max = 0
    for exp in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp)
        bar_max += models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp.experiment_type,
        ).count()
    bar = Bar('Processing', max=bar_max)

    for exp_1 in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp_1)
        for exp_2 in models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp_1.experiment_type,
        ):

            _exp_1, _exp_2 = sorted([exp_1, exp_2], key=lambda x: x.pk)
            rr_1 = _exp_1.experiment_type.relevant_regions
            rr_2 = _exp_2.experiment_type.relevant_regions
            if all([
                (_exp_1, _exp_2) not in updated,
                _exp_1 != _exp_2,
                models.PCATransformedValues.objects.filter(
                    dataset__experiment=_exp_1,
                    pca__locus_group__group_type=rr_1,
                ).exists(),
                models.PCATransformedValues.objects.filter(
                    dataset__experiment=_exp_2,
                    pca__locus_group__group_type=rr_2,
                ).exists(),
            ]):

                distance = score.score_experiments_by_pca_distance(
                    _exp_1, _exp_2)
                models.ExperimentDataDistance.objects.update_or_create(
                    experiment_1=_exp_1,
                    experiment_2=_exp_2,
                    defaults={
                        'distance': distance,
                    },
                )
                updated.add((_exp_1, _exp_2))

            bar.next()

    bar.finish()


@task
def update_experiment_metadata_scores(experiments):
    '''
    Update or create experiment metadata distance values.
    '''
    updated = set()

    bar_max = 0
    for exp in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp)
        bar_max += models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp.experiment_type,
        ).count()
    bar = Bar('Processing', max=bar_max)

    onts = []
    onts.append(models.Ontology.objects.get(name='cell_ontology'))
    onts.append(models.Ontology.objects.get(name='cell_line_ontology'))
    onts.append(models.Ontology.objects.get(name='brenda_tissue_ontology'))
    cell_ont_list = [ont.get_ontology_object() for ont in onts]

    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()

    for exp_1 in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp_1)
        for exp_2 in models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp_1.experiment_type,
        ):

            _exp_1, _exp_2 = sorted([exp_1, exp_2], key=lambda x: x.pk)
            if all([
                (_exp_1, _exp_2) not in updated,
                _exp_1 != _exp_2,
            ]):

                total_sim = 0

                cell_ont_sims = []
                for ont_obj in cell_ont_list:
                    sim = ont_obj.get_word_similarity(
                        _exp_1.cell_type, _exp_2.cell_type, metric='lin')
                    if sim:
                        cell_ont_sims.append(sim)
                if cell_ont_sims:
                    total_sim += max(cell_ont_sims)

                gene_ont_sim = gene_ont.get_word_similarity(
                    _exp_1.target, _exp_2.target, metric='jaccard',
                    weighting='information_content')
                if gene_ont_sim:
                    total_sim += gene_ont_sim

                models.ExperimentMetadataDistance.objects.update_or_create(
                    experiment_1=_exp_1,
                    experiment_2=_exp_2,
                    defaults={
                        'distance': total_sim,
                    },
                )
                updated.add((_exp_1, _exp_2))

            bar.next()

    bar.finish()
