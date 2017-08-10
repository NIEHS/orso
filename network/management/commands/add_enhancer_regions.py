from django.core.management.base import BaseCommand
from network import models


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('enhancer_list', type=str)
        '''
        Format of each annotation_list row:
        assembly path_to_enhancer_bed_file
        '''

    def handle(self, *args, **options):
        with open(options['enhancer_list']) as f:
            for line in f:
                assembly, enhancer_bed = line.strip().split()

                #  Retrieve assembly object
                assembly_obj = models.Assembly.objects.get(name=assembly)

                #  Create region object
                models.GenomicRegions.objects.create(
                    name=enhancer_bed.split('/')[-1].split('.bed')[0],
                    assembly=assembly_obj,
                    bed_file=enhancer_bed,
                    short_label='Enhancers',
                )
