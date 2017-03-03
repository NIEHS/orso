from django.core.management.base import BaseCommand
from network import models
from django.contrib.auth.models import User
import random
import string


def random_string():
    return ''.join(
        random.choice(string.ascii_uppercase + string.digits)
        for _ in range(6))


class Command(BaseCommand):
    def handle(self, *args, **options):
        experiments = list(models.Experiment.objects.all())

        for i in range(100):
            u = User.objects.create(
                username=random_string(),
                email=random_string() + '@gmail.com',
                password=random_string()
            )
            my_user = models.MyUser.objects.create(
                user=u,
                slug=u.username,
            )

            for exp in random.sample(experiments, 10):
                models.ExperimentFavorite.objects.create(
                    owner=my_user,
                    favorite=exp,
                )
