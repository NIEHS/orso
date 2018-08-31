import time

from celery.decorators import task
from django.db.models import Q

from network import models
from network.tasks import locks
from network.tasks.utils import run_tasks


def update_experiment_recommendations(experiments, **kwargs):
    tasks = []
    for exp in experiments:
        tasks.append(_update_experiment_recommendations.si(exp.pk, **kwargs))
    run_tasks(tasks, **kwargs)


@task
def _update_experiment_recommendations(
        experiment_pk, lock=True, sim_types=['metadata', 'primary'], **kwargs):

    experiment = models.Experiment.objects.get(pk=experiment_pk)
    users = models.MyUser.objects.filter(
        Q(favorite__experiment=experiment) |
        Q(experiment=experiment)
    ).distinct()

    if lock:
        execute_lock = locks.ExpRecUpdateExecuteLock(experiment)
        queue_lock = locks.ExpRecUpdateQueueLock(experiment)

        while execute_lock.exists():
            time.sleep(10)

        execute_lock.add()
        queue_lock.delete()

    # Remove old recs with old users
    models.Recommendation.objects.filter(
        referring_experiment=experiment,
    ).exclude(user__in=users).delete()

    # Remove old recs with no associated Similarity
    for rec_type in sim_types:
        for rec in models.Recommendation.objects.filter(
            referring_experiment=experiment,
            rec_type=rec_type,
        ):
            if not models.Similarity.objects.filter(
                sim_type=rec_type,
                experiment_1=experiment,
                experiment_2=rec.recommended_experiment,
            ).exists():
                rec.delete()

    # Add new recs
    for user in users:
        for sim in models.Similarity.objects.filter(
            experiment_1=experiment,
            sim_type__in=sim_types,
        ):
            models.Recommendation.objects.update_or_create(
                user=user,
                rec_type=sim.sim_type,
                referring_experiment=sim.experiment_1,
                referring_dataset=sim.dataset_1,
                recommended_experiment=sim.experiment_2,
                recommended_dataset=sim.dataset_2,
            )

    if lock:
        execute_lock.delete()


@task
def update_user_recommendations(following_user_pk, followed_user_pk,
                                lock=True):

    following = models.MyUser.objects.get(pk=following_user_pk)
    followed = models.MyUser.objects.get(pk=followed_user_pk)

    if lock:
        execute_lock = locks.UserRecUpdateExecuteLock(following, followed)
        queue_lock = locks.UserRecUpdateQueueLock(following, followed)

        while execute_lock.exists():
            time.sleep(10)

        execute_lock.add()
        queue_lock.delete()

    if models.Follow.objects.filter(
        following=following,
        followed=followed,
    ).exists():
        for exp in models.Experiment.objects.filter(owners=followed):
            models.Recommendation.objects.update_or_create(
                user=following,
                recommended_experiment=exp,
                referring_user=followed,
                rec_type='user',
            )
    else:
        try:
            models.Recommendation.objects.filter(
                user=following,
                referring_user=followed,
                rec_type='user',
            ).delete()
        except models.Recommendation.DoesNotExist:
            pass

    if lock:
        execute_lock.delete()
