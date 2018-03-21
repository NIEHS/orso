# from django.core.cache import cache
from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver

from . import tasks, models


@receiver(post_save, sender=models.Follow)
def follow_post_save_update(sender, instance, created, **kwargs):
    if created:
        lock = tasks.locks.UserRecUpdateQueueLock(
            instance.following, instance.followed)
        if lock.add():
            (tasks.recommendations
                  .update_user_recommendations
                  .si(instance.following.pk, instance.followed.pk).delay())


@receiver(pre_delete, sender=models.Follow)
def follow_pre_delete_update(sender, instance, **kwargs):
    lock = tasks.locks.UserRecUpdateQueueLock(
        instance.following, instance.followed)
    if lock.add():
        (tasks.recommendations
              .update_user_recommendations
              .si(instance.following.pk, instance.followed.pk).delay())


@receiver(post_save, sender=models.Favorite)
def favorite_post_save_update(sender, instance, created, **kwargs):
    if created:
        lock = tasks.locks.ExpRecUpdateQueueLock(instance.experiment)
        if lock.add():
            (tasks.recommendations
                  .update_recommendations
                  .si(instance.experiment.pk).delay())


@receiver(pre_delete, sender=models.Favorite)
def favorite_pre_delete_update(sender, instance, **kwargs):
    lock = tasks.locks.ExpRecUpdateQueueLock(instance.experiment)
    if lock.add():
        (tasks.recommendations
              .update_recommendations
              .si(instance.experiment.pk).delay())
