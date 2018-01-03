from django.core.management.base import BaseCommand

from network.tasks.normalize import normalize_dataset_intersections


HELP_TEXT = """Renormalize data intersection values."""


class Command(BaseCommand):
    help = HELP_TEXT

    def handle(self, *args, **options):
        normalize_dataset_intersections()
