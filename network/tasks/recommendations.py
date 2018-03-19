import time

from celery.decorators import task
from django.db.models import Q

from network import models
from network.tasks.locks import ExpRecUpdateExecuteLock as ExecuteLock
from network.tasks.locks import ExpRecUpdateQueueLock as QueueLock


@task
def update_recommendations(experiment_pk, lock=True):

    experiment = models.Experiment.objects.get(pk=experiment_pk)
    users = models.MyUser.objects.filter(
        Q(favorite__experiment=experiment) |
        Q(experiment=experiment)
    ).distinct()

    if lock:
        execute_lock = ExecuteLock(experiment)
        queue_lock = QueueLock(experiment)

        while execute_lock.exists():
            time.sleep(10)

        execute_lock.add()
        queue_lock.delete()

    # Remove old recs with old users
    models.Recommendation.objects.filter(
        referring_experiment=experiment,
    ).exclude(user__in=users).delete()

    # Remove old recs with no associated Similarity
    for rec_type in ['primary', 'metadata']:
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
        for sim in models.Similarity.objects.filter(experiment_1=experiment):
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
