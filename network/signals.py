from django.db.models.signals import post_save
from django.dispatch import receiver

from . import tasks, models


@receiver(post_save, sender=models.Dataset)
def trigger_dataset_processing(sender, instance, created, **kwargs):
    if created:
        tasks.process_dataset.delay(instance.id)
