from celery import group
from django.core.management.base import BaseCommand

from network import models
from network.management.commands.update_dendrogram import \
    call_update_dendrogram


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument(
            '--threads',
            action='store',
            dest='threads',
            type=int,
            help='Number of threads to use',
        )

    def handle(self, *args, **options):

        tasks = []

        for org in models.Organism.objects.all():
            for exp_type in models.ExperimentType.objects.all():

                experiments = models.Experiment.objects.filter(
                    dataset__assembly__organism=org,
                    experiment_type=exp_type,
                )

                if experiments.exists():
                    tasks.append(
                        call_update_dendrogram.si(org.pk, exp_type.pk))

                    for my_user in models.MyUser.objects.all():
                        if experiments.filter(owners=my_user).exists():
                            tasks.append(call_update_dendrogram.si(
                                org.pk, exp_type.pk, my_user_pk=my_user.pk))

        job = group(tasks)
        results = job.apply_async()
        results.join()
