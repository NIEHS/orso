import os

from django.core.management.base import BaseCommand

from network import models


class Command(BaseCommand):
    help = '''
        Add an ontology object.
    '''

    def add_arguments(self, parser):
        parser.add_argument('ontology_name', type=str)
        parser.add_argument('ontology_type', type=str)
        parser.add_argument('obo_file', type=str)
        parser.add_argument('ac_file', type=str)

    def handle(self, *args, **options):
        if models.Ontology.objects.filter(
                name=options['ontology_name']).exists():
            print(
                'Ontology object with name {} already exists. Exiting.'.format(
                    options['ontology_name']
                ))
        else:
            cwd = os.getcwd()
            obo_path = os.path.join(cwd, options['obo_file'])
            ac_path = os.path.join(cwd, options['ac_file'])
            models.Ontology.objects.create(
                name=options['ontology_name'],
                ontology_type=options['ontology_type'],
                obo_file=obo_path,
                ac_file=ac_path,
            )
