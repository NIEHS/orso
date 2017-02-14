from network import models
from django.contrib.auth.models import User


def random_string():
    return


class Command(BaseCommand):
    def handle(self, *args, **options):
        for i in range(100):
            u = User.objects.create(
                username=random_string(),
                email=random_string() + '@gmail.com',
                password=random_string()
            )
            models.MyUser.objects.create(
                user=u,
                slug=u.username,
            )
