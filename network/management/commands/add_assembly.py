from django.core.management.base import BaseCommand

from network import models


class Command(BaseCommand):
    help = '''
        Add an assembly object.
    '''

    def add_arguments(self, parser):
        parser.add_argument('assembly_name', type=str)
        parser.add_argument('chrom_sizes_path', type=str)

    def handle(self, *args, **options):
        if models.Assembly.objects.filter(
                name=options['assembly_name']).exists():
            print(
                'Assembly object with name {} already exists. Exiting.'.format(
                    options['assembly_name']
                ))
        else:
            assembly_obj = models.Assembly.objects.create(
                name=options['assembly_name'],
            )
            assembly_obj.read_in_chrom_sizes(options['chrom_sizes_path'])
