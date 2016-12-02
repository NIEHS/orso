from celery.decorators import task, periodic_task
from django.utils import timezone
from django.core.cache import cache

from analysis.metaplot import MetaPlot
from analysis.correlation import Correlation
from . import models

from datetime import timedelta
from collections import defaultdict
from functools import wraps


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


@periodic_task(run_every=timedelta(seconds=5))
@single_instance_task('data_recommendations')
def update_data_recommendations():
    def add_or_update_data_recommendations():
        # Find top 20 datasets not owned by user with highest Z score
        z_scores = models.CorrelationCell.get_z_score_list()

        # Only consider z_scores where one or more datasets is owned
        z_scores = [score for score in z_scores if score['users_1'] or score['users_2']]

        for user in models.MyUser.objects.all():

            user_score_dict = dict()
            for score in z_scores:
                in_1 = user.id in score['users_1']
                in_2 = user.id in score['users_2']
                if (in_1 or in_2) and not (in_1 and in_2):
                    if in_1:
                        recommended_id = score['dataset_2']
                    elif in_2:
                        recommended_id = score['dataset_1']
                    if not models.DataFavorite.objects.filter(
                            owner=user, favorite__id=recommended_id).exists():
                        if recommended_id in user_score_dict:
                            if score['max_z_score'] > \
                                    user_score_dict[recommended_id]['max_z_score']:
                                user_score_dict[recommended_id] = score
                        else:
                            user_score_dict[recommended_id] = score

            user_score_list = sorted(user_score_dict.values(), key=lambda x: -x['max_z_score'])

            for score in user_score_list[:20]:
                if user.id in score['users_1']:
                    reference_id = score['dataset_1']
                    recommended_id = score['dataset_2']
                elif user.id in score['users_2']:
                    reference_id = score['dataset_2']
                    recommended_id = score['dataset_1']
                recommended = models.Dataset.objects.get(id=recommended_id)
                reference = models.Dataset.objects.get(id=reference_id)

                dr = models.DataRecommendation.objects.filter(
                    owner=user,
                    recommended=recommended,
                    reference_dataset=reference,
                )
                df = models.DataFavorite.objects.filter(
                    owner=user,
                    favorite=recommended,
                )

                if not df.exists():
                    if dr.exists():
                        dr[0].score = score['max_z_score']
                        dr[0].save()
                    else:
                        models.DataRecommendation.objects.create(
                            owner=user,
                            score=score['max_z_score'],
                            recommended=recommended,
                            reference_dataset=reference,
                        )

    def clean_up_data_recommendations():
        time_threshold = timezone.now() - timedelta(days=7)
        models.DataRecommendation.objects.filter(last_updated__lt=time_threshold).delete()

    add_or_update_data_recommendations()
    clean_up_data_recommendations()


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
    for assembly in models.GenomeAssembly.objects.all():

        datasets = models.Dataset.objects.filter(
            assembly=assembly,
            promoter_intersection__isnull=False,
            enhancer_intersection__isnull=False,
        ).order_by('id')
        annotation = assembly.default_annotation

        for regions in [annotation.promoters, annotation.enhancers]:
            for i, ds_1 in enumerate(datasets):
                for j, ds_2 in enumerate(datasets[i + 1:]):
                    if not (models.CorrelationCell.objects
                            .filter(x_dataset=ds_1, y_dataset=ds_2, genomic_regions=regions)
                            .exists()):

                        if regions == annotation.promoters:
                            corr = Correlation(
                                ds_1.promoter_intersection.intersection_values,
                                ds_2.promoter_intersection.intersection_values,
                            ).get_correlation()[0]
                        elif regions == annotation.enhancers:
                            corr = Correlation(
                                ds_1.enhancer_intersection.intersection_values,
                                ds_2.enhancer_intersection.intersection_values,
                            ).get_correlation()[0]

                        models.CorrelationCell.objects.create(
                            x_dataset=ds_1,
                            y_dataset=ds_2,
                            genomic_regions=regions,
                            score=corr,
                        )
