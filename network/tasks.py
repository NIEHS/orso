from celery.decorators import task, periodic_task
from django.utils import timezone
from django.core.cache import cache

from analysis.metaplot import MetaPlot
from analysis.correlation import Correlation
from . import models

from datetime import timedelta
from collections import defaultdict
from functools import wraps

import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import jaccard_similarity_score


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
def process_dataset(id_):
    dataset = models.Dataset.objects.get(id=id_)

    promoter_regions = dataset.assembly.default_annotation.promoters
    enhancer_regions = dataset.assembly.default_annotation.enhancers

    pm = MetaPlot(
        promoter_regions.bed_file.path,
        single_bw=dataset.ambiguous_url
    )

    em = MetaPlot(
        enhancer_regions.bed_file.path,
        single_bw=dataset.ambiguous_url,
    )

    dataset.promoter_metaplot = models.MetaPlot.objects.create(
        genomic_regions=promoter_regions,
        bigwig_url=dataset.ambiguous_url,
        relative_start=-2500,
        relative_end=2499,
        meta_plot=pm.create_metaplot_json(),
    )
    dataset.enhancer_metaplot = models.MetaPlot.objects.create(
        genomic_regions=enhancer_regions,
        bigwig_url=dataset.ambiguous_url,
        relative_start=-2500,
        relative_end=2499,
        meta_plot=em.create_metaplot_json(),
    )
    dataset.promoter_intersection = models.IntersectionValues.objects.create(
        genomic_regions=promoter_regions,
        bigwig_url=dataset.ambiguous_url,
        relative_start=-2500,
        relative_end=2499,
        intersection_values=pm.create_intersection_json(),
    )
    dataset.enhancer_intersection = models.IntersectionValues.objects.create(
        genomic_regions=enhancer_regions,
        bigwig_url=dataset.ambiguous_url,
        relative_start=-2500,
        relative_end=2499,
        intersection_values=em.create_intersection_json(),
    )

    dataset.save()


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
                'collaborative_experiment': None,
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
                recommended=exp,
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
        models.UserRecommendation.objects.filter(last_updated__lt=time_threshold).delete()

    add_or_update_user_recommendations()
    clean_up_user_recommendations()


@periodic_task(run_every=timedelta(seconds=5))
@single_instance_task('correlation_values')
def update_correlation_values():
    experiments = models.Experiment.objects.all().order_by('id')

    for i, exp_1 in enumerate(experiments):
        intersections_1 = exp_1.get_average_intersections()
        for j, exp_2 in enumerate(experiments[i:]):
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
