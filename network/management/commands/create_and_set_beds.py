from django.core.management.base import BaseCommand

from network import models


class Command(BaseCommand):

    def handle(self, *args, **options):
        for lg in models.LocusGroup.objects.all():
            lg.create_and_set_metaplot_bed()
            lg.create_and_set_intersection_bed()
