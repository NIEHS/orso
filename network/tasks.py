from celery.decorators import task, periodic_task
from celery import group
from django.utils import timezone
from django.core.cache import cache

from analysis.correlation import Correlation
from analysis import transcript_coverage
from . import models

from datetime import timedelta
from collections import defaultdict
from functools import wraps

import string
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import jaccard_similarity_score
from sklearn.decomposition import PCA

from combat.combat import combat
import pandas as pd

import os
from tempfile import NamedTemporaryFile
import json
from analysis import metaplot
from analysis.normalization import normalize_transcript_intersection_values
from analysis.expression import select_transcript_by_expression

from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import RandomForestClassifier

from analysis import score


def single_instance_task(cache_id, timeout=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if cache.add(cache_id, True, timeout):
                try:
                    func(*args, **kwargs)
                finally:
                    cache.delete(cache_id)
        return wrapper
    return decorator


@task()
def select_all_representative_transcripts(genes):
    job = group(_select_representative_transcripts.s(g.pk) for g in genes)
    job.apply_async()


@task()
def _select_representative_transcripts(gene_pk):
    select_transcript_by_expression(models.Gene.objects.get(pk=gene_pk))


@task()
def process_datasets(datasets):

    download_list_file = NamedTemporaryFile(mode='w')
    bigwig_paths = dict()
    for ds in datasets:
        if ds.is_stranded():
            download_list_file.write('{}\n'.format(ds.plus_url))
            download_list_file.write('{}\n'.format(ds.minus_url))
            bigwig_paths[ds.pk] = {
                'ambiguous': None,
                'plus': os.path.basename(ds.plus_url),
                'minus': os.path.basename(ds.minus_url),
            }
        else:
            download_list_file.write('{}\n'.format(ds.ambiguous_url))
            bigwig_paths[ds.pk] = {
                'ambiguous': os.path.basename(ds.ambiguous_url),
                'plus': None,
                'minus': None,
            }
    download_list_file.flush()
    call([
        'aria2c',
        '--allow-overwrite=true',
        '--conditional-get=true',
        '-x', '16',
        '-s', '16',
        '-i', download_list_file.name,
    ])
    download_list_file.close()

    assembly_to_intersection_bed = dict()
    for ds in datasets:
        if ds.assembly not in assembly_to_intersection_bed:
            assembly_to_intersection_bed[ds.assembly] = \
                NamedTemporaryFile(mode='w', delete=False)
            transcript_coverage.generate_transcript_bed(
                ds.assembly.get_transcripts(),
                json.loads(ds.assembly.chromosome_sizes),
                assembly_to_intersection_bed[ds.assembly],
            )

    gr_to_metaplot_bed = dict()
    for ds in datasets:
        for gr in models.GenomicRegions.objects.filter(assembly=ds.assembly):
            if gr not in gr_to_metaplot_bed:
                gr_to_metaplot_bed[gr] = \
                    NamedTemporaryFile(mode='w', delete=False)
                metaplot.generate_metaplot_bed(
                    gr,
                    json.loads(gr.assembly.chromosome_sizes),
                    gr_to_metaplot_bed[gr],
                )

    tasks = []
    for ds in datasets:
        tasks.append(process_dataset_intersection.s(
            ds.pk,
            assembly_to_intersection_bed[ds.assembly].name,
            bigwig_paths[ds.pk],
        ))

        for gr in models.GenomicRegions.objects.filter(assembly=ds.assembly):
            tasks.append(process_dataset_metaplot.s(
                ds.pk,
                gr.pk,
                gr_to_metaplot_bed[gr].name,
                bigwig_paths[ds.pk],
            ))

    job = group(tasks)
    results = job.apply_async()
    results.join()

    for temp_bed in assembly_to_intersection_bed.values():
        temp_bed.close()

    for temp_bed in gr_to_metaplot_bed.values():
        temp_bed.close()


@task()
def process_dataset_intersection(dataset_pk, transcript_bed, bigwigs):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    transcripts = dataset.assembly.get_transcripts()
    transcript_values = transcript_coverage.get_transcript_values(
        transcripts,
        transcript_bed,
        ambiguous_bigwig=bigwigs['ambiguous'],
        plus_bigwig=bigwigs['plus'],
        minus_bigwig=bigwigs['minus'],
    )
    normalized_values = \
        normalize_transcript_intersection_values(transcript_values)
    models.TranscriptIntersection.objects.filter(dataset=dataset).delete()
    entries = []
    for transcript, value_dict in transcript_values.items():
        norm = normalized_values[transcript]
        entries.append(models.TranscriptIntersection(
            dataset=dataset,
            transcript=transcript,
            promoter_value=value_dict['promoter'],
            genebody_value=value_dict['genebody'],
            exon_values=value_dict['exons'],
            intron_values=value_dict['introns'],
            normalized_promoter_value=norm['promoter'],
            normalized_genebody_value=norm['genebody'],
            normalized_coding_value=norm['coding'],
        ))
    models.TranscriptIntersection.objects.bulk_create(entries)


@task()
def process_dataset_metaplot(dataset_pk, gr_pk, bed_path, bigwigs):

    dataset = models.Dataset.objects.get(pk=dataset_pk)
    genomic_regions = models.GenomicRegions.objects.get(pk=gr_pk)

    metaplot_values, intersection_values = metaplot.get_metaplot_values(
        genomic_regions,
        bed_path=bed_path,
        ambiguous_bigwig=bigwigs['ambiguous'],
        plus_bigwig=bigwigs['plus'],
        minus_bigwig=bigwigs['minus'],
    )

    models.MetaPlot.objects.update_or_create(
        dataset=dataset,
        genomic_regions=genomic_regions,
        defaults={
            'metaplot': json.dumps(metaplot_values),
        }
    )
    models.IntersectionValues.objects.update_or_create(
        dataset=dataset,
        genomic_regions=genomic_regions,
        defaults={
            'intersection_values': json.dumps(intersection_values),
        }
    )


def get_metadata_similarities():
    EXPERIMENT_DESCRIPTION_FIELDS = [
        'assay_slims',
        'assay_synonyms',
        'assay_term_name',
        'assay_title',
        'biosample_summary',
        'biosample_synonyms',
        'biosample_term_name',
        'biosample_type',
        'category_slims',
        'objective_slims',
        'organ_slims',
        'target',
        'system_slims',
    ]

    descriptions = []
    exps = []

    for exp in models.Experiment.objects.all():
        description = exp.description

        description = description.translate(
            str.maketrans('', '', string.punctuation))
        description = description.split()
        for field in EXPERIMENT_DESCRIPTION_FIELDS:
            description = [x for x in description if x != field]
        if exp.data_type:
            description.append(exp.data_type)
        if exp.cell_type:
            description.append(exp.cell_type)
        if exp.target:
            description.append(exp.target)

        descriptions.append(' '.join(description))
        exps.append(exp)

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0,
                         stop_words='english')
    tfidf_matrix = tf.fit_transform(descriptions)
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    similarities = dict()
    for i, row in enumerate(cosine_similarities):
        similarities[exps[i]] = dict()
        for j, value in enumerate(row):
            if i != j:
                similarities[exps[i]][exps[j]] = value
    return similarities


def get_region_variance(gr):
    intersection_values = models.IntersectionValues.objects.filter(
        genomic_regions=gr)

    if intersection_values:

        #  Populate intersection matrix
        intersection_matrix = []
        for iv in intersection_values:
            intersection_matrix.append(iv.intersection_values)

        #  Normalize by sum
        for i, row in enumerate(intersection_matrix):
            _sum = sum(row)
            norm_row = []
            for entry in row:
                norm_row.append(entry / _sum)
            intersection_matrix[i] = norm_row

        #  Calculate variance
        variance = numpy.var(intersection_matrix, 0)

        #  Calculate variance mask
        n = int(0.1 * len(variance))  # Find top 10%; min of 200, max of 1000
        if n < 200:
            n = 200
        elif n > 1000:
            n = 1000
        cutoff = sorted(variance, reverse=True)[n - 1]

        variance_mask = []
        for var in variance:
            if var >= cutoff:
                variance_mask.append(1)
            else:
                variance_mask.append(0)

        #  Update GR model
        gr.variance = list(variance)
        gr.variance_mask = variance_mask
        gr.save()


def get_user_similarites():
    experiments = models.Experiment.objects.all()
    users = models.MyUser.objects.all()
    user_vectors = dict()
    similarities = dict()

    for user in users:
        user_vectors[user] = []
        for exp in experiments:
            if models.ExperimentFavorite.objects.filter(
                    owner=user, favorite=exp).exists():
                user_vectors[user].append(1)
            else:
                user_vectors[user].append(0)

    for user_1 in users:
        similarities[user_1] = dict()
        for user_2 in users:
            similarities[user_1][user_2] = jaccard_similarity_score(
                user_vectors[user_1],
                user_vectors[user_2],
            )

    return similarities


@periodic_task(run_every=timedelta(seconds=5))
@single_instance_task('data_recommendations')
def update_data_recommendations():
    users = models.MyUser.objects.all()
    z_scores = models.ExperimentCorrelation.get_max_z_scores()
    metadata_similarities = get_metadata_similarities()
    user_similarities = get_user_similarites()

    for user in users:
        recs = dict()
        owned_exps = models.Experiment.objects.filter(owners__in=[user])

        #  Find genomic regions where user owns an associated experiment
        user_grs = set()
        for intersection in models.IntersectionValues.objects.filter(
                dataset__experiment__owners__in=[user]):
            user_grs.add(intersection.genomic_regions)

        #  Find all potential experiments for comparison
        compare_exps = set()
        for gr in user_grs:
            for intersection in (models.IntersectionValues.objects
                                 .filter(genomic_regions=gr)
                                 .exclude(dataset__experiment__owners__in=[user])):  # noqa
                compare_exps.add(intersection.dataset.experiment)
        #  Populate rec dict with all potential experiments
        for exp in compare_exps:
            recs[exp] = {
                'correlation_rank': None,
                'correlation_experiment': None,
                'metadata_rank': None,
                'metadata_experiment': None,
                'collaborative_rank': None,
            }

        #  Find dataset correlation rank order
        #  Get z scores for each experiment
        z_score_dict = defaultdict(list)
        for exps, score in z_scores.items():
            exp_1, exp_2 = exps
            if exp_1 in owned_exps and exp_2 in compare_exps:
                z_score_dict[exp_2].append([score, exp_1])
            elif exp_1 in compare_exps and exp_2 in owned_exps:
                z_score_dict[exp_1].append([score, exp_2])

        #  Find max z score for each experiment
        max_z_score_dict = dict()
        for exp, scores in z_score_dict.items():
            _sorted = sorted(scores, key=lambda x: (-x[0], x[1]))
            max_z_score_dict[exp] = {
                'max_z_score': _sorted[0][0],
                'exp': _sorted[0][1],
            }

        #  Get rank from sorted list
        for i, entry in enumerate(sorted(
            max_z_score_dict.items(),
            key=lambda x: (-x[1]['max_z_score'], x[1]['exp'].name))
        ):
            recs[entry[0]]['correlation_rank'] = i
            recs[entry[0]]['correlation_experiment'] = entry[1]['exp']

        #  Find metadata comparison rank order
        #  Get similarities for each experiment
        metadata_sim_dict = defaultdict(list)
        for exp_1 in compare_exps:
            for exp_2 in owned_exps:
                metadata_sim_dict[exp_1].append(
                    (metadata_similarities[exp_1][exp_2], exp_2))

        #  Find max similarities
        max_sim_dict = dict()
        for exp, similarities in metadata_sim_dict.items():
            _sorted = sorted(similarities, key=lambda x: (-x[0], x[1].name))
            max_sim_dict[exp] = {
                'max_sim': _sorted[0][0],
                'exp': _sorted[0][1],
            }

        #  Get rank from sorted list
        for i, entry in enumerate(sorted(
            max_sim_dict.items(),
            key=lambda x: (-x[1]['max_sim'], x[1]['exp'].name))
        ):
            recs[entry[0]]['metadata_rank'] = i
            recs[entry[0]]['metadata_experiment'] = entry[1]['exp']

        #  Find user comparison rank order
        #  Get weighted scores for each experiment
        user_scores = defaultdict(list)
        for exp in compare_exps:
            for comp_user in users:
                if models.ExperimentFavorite.objects.filter().exists():
                    score = 1 * user_similarities[user][comp_user]
                else:
                    score = 0
                user_scores[exp].append(score)

        #  Sum weighted scores for each experiment
        total_user_score = dict()
        for exp, scores in user_scores.items():
            total_user_score[exp] = sum(scores)

        #  Get rank from sorted list
        for i, entry in enumerate(sorted(
            max_sim_dict.items(),
            key=lambda x: -x[1]['max_sim']
        )):
            recs[entry[0]]['collaborative_rank'] = i

        for rec, ranks in recs.items():
            models.ExperimentRecommendation.objects.update_or_create(
                owner=user,
                recommended=rec,
                defaults=ranks,
            )


@periodic_task(run_every=timedelta(seconds=5))
@single_instance_task('user_recommendations')
def update_user_recommendations():
    def add_or_update_user_recommendations():
        # Add a user recommendation if a user's dataset has been favorited
        # Score is the number of favorited dataset_counts
        scores = defaultdict(int)

        for fav in models.DataFavorite.objects.all():
            for dataset_owner in fav.favorite.owners.all():
                if fav.owner != dataset_owner:
                    scores[(fav.owner, dataset_owner)] += 1

        for key, score in scores.items():
            owner, favorite = key

            ur = models.UserRecommendation.objects.filter(
                owner=owner,
                recommended=favorite,
            )
            uf = models.UserFavorite.objects.filter(
                owner=owner,
                favorite=favorite,
            )

            if not uf.exists():
                if ur.exists():
                    ur[0].score = score
                    ur[0].save()
                else:
                    models.UserRecommendation.objects.create(
                        owner=owner,
                        recommended=favorite,
                        score=score,
                    )

    def clean_up_user_recommendations():
        time_threshold = timezone.now() - timedelta(days=7)
        models.UserRecommendation.objects.filter(
            last_updated__lt=time_threshold).delete()

    add_or_update_user_recommendations()
    clean_up_user_recommendations()


@periodic_task(run_every=timedelta(seconds=5))
@single_instance_task('correlation_values')
def update_correlation_values():
    experiments = models.Experiment.objects.all().order_by('id')

    for i, exp_1 in enumerate(experiments):
        intersections_1 = exp_1.get_average_intersections()
        for j, exp_2 in enumerate(experiments[i:]):

            if exp_1 != exp_2:

                intersections_2 = exp_2.get_average_intersections()

                for int_1 in intersections_1:
                    for int_2 in intersections_2:
                        if int_1['regions_pk'] == int_2['regions_pk']:
                            gr = models.GenomicRegions.objects.get(
                                pk=int_1['regions_pk'])
                            if not (models.ExperimentCorrelation.objects
                                    .filter(
                                        x_experiment=exp_1,
                                        y_experiment=exp_2,
                                        genomic_regions=gr,
                                    ).exists()):
                                corr = Correlation(
                                    int_1['intersection_values'],
                                    int_2['intersection_values'],
                                ).get_correlation()[0]
                                models.ExperimentCorrelation.objects.create(
                                    x_experiment=exp_1,
                                    y_experiment=exp_2,
                                    genomic_regions=gr,
                                    score=corr,
                                )


@periodic_task(run_every=timedelta(seconds=5))
@single_instance_task('correlation_values')
def update_metadata_correlation_values():
    experiments = models.Experiment.objects.all().order_by('id')
    similarities = get_metadata_similarities()
    for i, exp_1 in enumerate(experiments):
        for j, exp_2 in enumerate(experiments[i + 1:]):
            models.MetadataCorrelation.objects.update_or_create(
                x_experiment=exp_1,
                y_experiment=exp_2,
                score=similarities[exp_1][exp_2],
            )


@task
def _get_associated_transcript(gene_pk):
    gene = models.Gene.objects.get(pk=gene_pk)
    return gene.get_transcript_with_highest_expression()


@task
def pca_analysis(annotation):

    def run_in_parallel(obj_list, task):
        res = group(task.s(obj.pk) for obj in obj_list)()
        return res.get()

    genes = models.Gene.objects.filter(annotation=annotation)[:100]
    transcripts = run_in_parallel(genes, _get_associated_transcript)
    transcripts = [t for t in transcripts if t is not None]

    for exp_type in models.ExperimentType.objects.all():

        datasets = models.Dataset.objects.filter(
            assembly__annotation=annotation,
            experiment__experiment_type=exp_type,
        )

        if datasets:
            pca = PCA(n_components=3)
            rf = RandomForestClassifier(n_estimators=1000)

            intersection_values = dict()
            experiment_pks = dict()
            cell_types = dict()
            targets = dict()

            for ds in datasets:

                exp = models.Experiment.objects.get(dataset=ds)
                experiment_pks[ds] = exp.pk
                cell_types[ds] = exp.cell_type
                targets[ds] = exp.target

                intersection_values[ds] = dict()

                transcripts = sorted(transcripts, key=lambda x: x.pk)
                intersections = (
                    models.TranscriptIntersection
                          .objects.filter(dataset=ds,
                                          transcript__in=transcripts)
                          .order_by('transcript__pk')
                )

                for transcript, intersection in zip(
                        transcripts, intersections):
                    if exp_type.relevant_regions == 'genebody':
                        intersection_values[ds][transcript] = \
                            intersection.normalized_genebody_value
                    elif exp_type.relevant_regions == 'coding':
                        intersection_values[ds][transcript] = \
                            intersection.normalized_coding_value
                    elif exp_type.relevant_regions == 'promoter':
                        intersection_values[ds][transcript] = \
                            intersection.normalized_promoter_value

            # Filter transcripts by RF importance
            _intersection_values = []
            _cell_types = []
            _targets = []

            for ds in datasets:
                _intersection_values.append([])
                for ts in transcripts:
                    _intersection_values[-1].append(
                        intersection_values[ds][ts])
                _cell_types.append(cell_types[ds])
                _targets.append(targets[ds])

            cell_type_importances = rf.fit(
                _intersection_values, _cell_types).feature_importances_
            target_importances = rf.fit(
                _intersection_values, _cell_types).feature_importances_
            totals = [x + y for x, y in zip(cell_type_importances,
                                            target_importances)]
            filtered_transcripts = \
                [ts for ts, total in sorted(zip(transcripts, totals),
                                            key=lambda x: -x[1])]

            # Filter datasets by Mahalanobis distance after PCA
            filtered_datasets = []
            if len(datasets) >= 10:
                _intersection_values = []
                for ds in datasets:
                    _intersection_values.append([])
                    for ts in filtered_transcripts:
                        _intersection_values[-1].append(
                            intersection_values[ds][ts])

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
                for ts in filtered_transcripts:
                    _intersection_values[-1].append(
                        intersection_values[ds][ts])
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
                for ts in filtered_transcripts:
                    _intersection_values.append(
                        intersection_values[ds][ts])
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
                annotation=annotation,
                experiment_type=exp_type,
                defaults={
                    'plot': pca_plot,
                    'pca': pca,
                    'covariation_matrix': cov,
                    'inverse_covariation_matrix': inv,
                },
            )

            # Set the PCA to transcripts relationship
            if not created:
                pca.selected_transcripts.clear()
            _pca_transcript_orders = []
            for i, transcript in enumerate(filtered_transcripts):
                _pca_transcript_orders.append(models.PCATranscriptOrder(
                    pca=pca,
                    transcript=transcript,
                    order=i,
                ))
            models.PCATranscriptOrder.objects.bulk_create(
                _pca_transcript_orders)

            # Set the PCA to datasets relationship
            if not created:
                pca.transformed_datasets.clear()
            _pca_transformed_datasets = []
            for ds, values in zip(datasets, fitted_values):
                _pca_transformed_datasets.append(models.PCATransformedValues(
                    pca=pca,
                    dataset=ds,
                    transformed_values=values.tolist(),
                ))
            models.PCATransformedValues.objects.bulk_create(
                _pca_transformed_datasets)


@task
def transform_dataset_values_by_pca(datasets):
    '''
    For datasets, transform values and set PCATransformedValues.
    '''
    for ds in datasets:
        pca = models.PCA.objects.get(
            annotation__assembly=ds.assembly,
            experiment_type=ds.experiment.experiment_type,
        )
        fitted_values = transform.pca_transform_intersections(ds)
        models.PCATransformedValues.objects.update_or_create(
            pca=pca,
            dataset=ds,
            transformed_values=fitted_values.tolist(),
        )


@task
def get_tfidf_vectorizers():
    '''
    For each annotation/experiment type pair, create a TF/IDF vectorizer and
    create the associated object.
    '''
    EXPERIMENT_DESCRIPTION_FIELDS = [
        'assay_slims',
        'assay_synonyms',
        'assay_term_name',
        'assay_title',
        'biosample_summary',
        'biosample_synonyms',
        'biosample_term_name',
        'biosample_type',
        'category_slims',
        'objective_slims',
        'organ_slims',
        'target',
        'system_slims',
    ]

    for annotation in models.Annotation.objects.all():
        for exp_type in models.ExperimentType.objects.all():

            descriptions = []
            experiments = models.Experiment.objects.filter(
                dataset__assembly__annotation=annotation,
                experiment_type=exp_type,
            )

            if experiments:
                for exp in experiments:

                    description = exp.description
                    description = description.translate(
                        str.maketrans('', '', string.punctuation))
                    description = description.split()
                    for field in EXPERIMENT_DESCRIPTION_FIELDS:
                        description = [x for x in description if x != field]

                    if exp.cell_type:
                        description.append(exp.cell_type)
                    if exp.target:
                        description.append(exp.target)

                    descriptions.append(' '.join(description))

                tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),
                                     min_df=0, stop_words='english')
                tf.fit(descriptions)

                models.TfidfVectorizer.objects.update_or_create(
                    annotation=annotation,
                    experiment_type=exp_type,
                    defaults={
                        'tfidf_vectorizer': tf,
                    },
                )


@task
def set_dataset_data_distance(pk_1, pk_2):
    '''
    For the datasets, calculate and set the DatasetDataDistance.
    '''
    dataset_1 = models.Dataset.objects.get(pk=pk_1)
    dataset_2 = models.Dataset.objects.get(pk=pk_2)
    distance = score.score_datasets_by_pca_distance(dataset_1, dataset_2)
    models.DatasetDataDistance.objects.update_or_create(
        dataset_1=dataset_1,
        dataset_2=dataset_2,
        defaults={
            'distance': distance,
        }
    )


@task
def set_dataset_metadata_distance(pk_1, pk_2):
    '''
    For the datasets, calculate and set the DatasetMetadataDistance.
    '''
    dataset_1 = models.Dataset.objects.get(pk=pk_1)
    dataset_2 = models.Dataset.objects.get(pk=pk_2)
    distance = score.score_datasets_by_tfidf(dataset_1, dataset_2)
    models.DatasetMetadataDistance.objects.update_or_create(
        dataset_1=dataset_1,
        dataset_2=dataset_2,
        defaults={
            'distance': distance,
        }
    )


@task
def set_experiment_data_distance(pk_1, pk_2):
    '''
    For the experiments, calculate and set the ExperimentDataDistance.
    '''
    experiment_1 = models.Experiment.objects.get(pk=pk_1)
    experiment_2 = models.Experiment.objects.get(pk=pk_2)
    distance = score.score_experiments_by_pca_distance(
        experiment_1, experiment_2)
    models.ExperimentDataDistance.objects.update_or_create(
        experiment_1=experiment_1,
        experiment_2=experiment_2,
        defaults={
            'distance': distance,
        }
    )


@task
def set_experiment_metadata_distance(pk_1, pk_2):
    '''
    For the experiments, calculate and set the ExperimentMetadataDistance.
    '''
    experiment_1 = models.Experiment.objects.get(pk=pk_1)
    experiment_2 = models.Experiment.objects.get(pk=pk_2)
    distance = score.score_experiments_by_tfidf(experiment_1, experiment_2)
    models.ExperimentMetadataDistance.objects.update_or_create(
        experiment_1=experiment_1,
        experiment_2=experiment_2,
        defaults={
            'distance': distance,
        }
    )


@task
def set_dataset_distances(datasets):
    '''
    For all datasets, set DatasetDataDistance and DatasetMetadataDistance.
    '''
    other_datasets = models.Dataset.objects.exclude(
        pcatransformedvalues__isnull=True)
    for ds_1 in datasets:
        for ds_2 in other_datasets:
            if ds_1 != ds_2:
                if all([
                    ds_1.assembly == ds_2.assembly,
                    ds_1.experiment.experiment_type ==
                    ds_2.experiment.experiment_type,
                ]):
                    set_dataset_data_distance.delay(ds_1.pk, ds_2.pk)
                    set_dataset_metadata_distance.delay(ds_1.pk, ds_2.pk)


@task
def set_experiment_distances(experiments):
    '''
    For all experiments, set ExperimentDataDistance and
    ExperimentMetadataDistance.
    '''
    other_experiments = models.Experiment.objects.exclude(
        dataset__pcatransformedvalues__isnull=True)
    for exp_1 in experiments:
        for exp_2 in other_experiments:
            if exp_1 != exp_2:
                assemblies_1 = models.Assembly.objects.filter(
                    dataset__experiment=exp_1)
                assemblies_2 = models.Assembly.objects.filter(
                    dataset__experiment=exp_2)
                if all([
                    assemblies_1 & assemblies_2,
                    exp_1.experiment_type == exp_2.experiment_type,
                ]):
                    set_experiment_data_distance.delay(exp_1.pk, exp_2.pk)
                    set_experiment_metadata_distance.delay(exp_1.pk, exp_2.pk)
