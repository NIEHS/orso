from django.core.management.base import BaseCommand

from network import models

ENCODE_BASE_URL = 'https://www.encodeproject.org/experiments/{}/'
IHEC_BASE_URL = 'https://epigenomesportal.ca/ihec/grid.html?assembly={}'

IHEC_PROJECTS = [
    'DEEP',
    'CEEHRC',
    'KNIH',
    'Blueprint',
    'GIS',
    'AMED-CREST',
]
IHEC_ASSEMBLY_NAME_TO_ID = {
    'hg19': '1',
    'GRCh38': '4',
    'mm10': '3',
}


class Command(BaseCommand):
    help = '''
        Populate the consortial_url field for project experiments.
    '''

    def handle(self, *args, **options):

        # ENCODE
        for exp in models.Experiment.objects.filter(
                project__name='ENCODE'):
            if exp.consortial_id:
                exp.consortial_url = ENCODE_BASE_URL.format(exp.consortial_id)
                exp.save()

        # IHEC
        for exp in models.Experiment.objects.filter(
                project__name__in=IHEC_PROJECTS):
            if exp.consortial_id:
                ds = models.Dataset.objects.filter(experiment=exp)[0]
                assembly = ds.assembly.name
                assembly_id = IHEC_ASSEMBLY_NAME_TO_ID[assembly]
                exp.consortial_url = IHEC_BASE_URL.format(assembly_id)
                exp.save()
