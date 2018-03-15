from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver

from . import tasks, models


# TODO: add a lock for favorite recommendation updates; can be queued/fired
# multiple times

@receiver(post_save, sender=models.Favorite)
def favorite_post_save_update(sender, instance, created, **kwargs):
    if created:
        (tasks.metadata_recommendations
              .update_metadata_recommendations
              .si(instance.experiment.pk).delay())
        (tasks.data_recommendations
              .update_primary_data_recommendations
              .si(instance.experiment.pk).delay())


@receiver(pre_delete, sender=models.Favorite)
def favorite_pre_delete_update(sender, instance, **kwargs):
    (tasks.metadata_recommendations
          .update_metadata_recommendations
          .si(instance.experiment.pk).delay())
    (tasks.data_recommendations
          .update_primary_data_recommendations
          .si(instance.experiment.pk).delay())
