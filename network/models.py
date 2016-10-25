import requests

from django.db import models
from django.conf import settings
from django.contrib.postgres.fields import JSONField, ArrayField


class MyUser(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL)
    slug = models.CharField(
        max_length=128)

    favorite_users = models.ManyToManyField(
        'MyUser', symmetrical=False, blank=True)
    favorite_data = models.ManyToManyField(
        'Dataset', blank=True)

    def get_user_favorites_count(self):
        pass

    def get_data_favorites_count(self):
        pass


class Dataset(models.Model):
    DATA_TYPES = (
        ('Cage', 'Cage'),
        ('ChiaPet', 'ChiaPet'),
        ('ChipSeq', 'ChipSeq'),
        ('DnaseDgf', 'DnaseDgf'),
        ('DnaseSeq', 'DnaseSeq'),
        ('FaireSeq', 'FaireSeq'),
        ('Mapability', 'Mapability'),
        ('Nucleosome', 'Nucleosome'),
        ('Orchid', 'Orchid'),
        ('RepliChip', 'RepliChip'),
        ('RepliSeq', 'RepliSeq'),
        ('RipSeq', 'RipSeq'),
        ('RnaPet', 'RnaPet'),
        ('RnaSeq', 'RnaSeq'),
        ('StartSeq', 'StartSeq'),
        ('Other', 'Other (describe in "description" field)'),
    )

    data_type = models.CharField(
        max_length=16,
        choices=DATA_TYPES)
    cell_type = models.CharField(max_length=128)
    antibody = models.CharField(max_length=128)

    description = models.TextField(blank=True)

    ambiguous_url = models.URLField()
    plus_url = models.URLField()
    minus_url = models.URLField()

    owners = models.ManyToManyField('MyUser')
    name = models.CharField(max_length=128)
    slug = models.CharField(
        max_length=128)
    intersection_values = JSONField()
    created = models.DateTimeField(
        auto_now_add=True)

    promoter_metaplot = \
        models.ForeignKey('MetaPlot', related_name='promoter_meta')
    enhancer_metaplot = \
        models.ForeignKey('MetaPlot', related_name='enhancer_meta')

    def get_favorites_count(self):
        pass

    @staticmethod
    def check_valid_url(url):
        # ensure URL is valid and doesn't raise a 400/500 error
        try:
            resp = requests.head(url)
        except requests.exceptions.ConnectionError:
            return False, '{} not found.'.format(url)
        else:
            return resp.ok, '{}: {}'.format(resp.status_code, resp.reason)


class MetaPlot(models.Model):
    genomic_regions = models.ForeignKey('GenomicRegions')

    relative_start = models.IntegerField()
    relative_end = models.IntegerField()
    meta_plot = JSONField()
    last_updated = models.DateTimeField(
        auto_now=True)

    class Meta:
        unique_together = (
            ('genomic_regions', 'relative_start', 'relative_end',),
        )


class Recommendation(models.Model):  # Add something to sort
    owner = models.ForeignKey('MyUser')
    last_updated = models.DateTimeField(
        auto_now=True)
    score = models.FloatField()

    class Meta:
        abstract = True
        ordering = ('score', '-last_updated',)


class UserRecommendation(Recommendation):
    recommended = models.ForeignKey('MyUser', related_name='recommended')


class DataRecommendation(Recommendation):
    recommended = models.ForeignKey('Dataset')


class CorrelationCell(models.Model):
    x_dataset = models.ForeignKey('Dataset', related_name='x')
    y_dataset = models.ForeignKey('Dataset', related_name='y')
    score = models.FloatField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (
            ('x_dataset', 'y_dataset',),
        )

    @classmethod
    def as_matrix(cls):
        pass


class GenomeAssembly(models.Model):
    name = models.CharField(
        unique=True,
        max_length=32)
    default_annotation = models.ForeignKey(
        'GeneAnnotation',
        blank=True,
        null=True)
    last_updated = models.DateTimeField(
        auto_now=True)


class GeneAnnotation(models.Model):
    name = models.CharField(
        max_length=32)
    assembly = models.ForeignKey('GenomeAssembly')
    gtf_file = models.FileField()
    last_updated = models.DateTimeField(
        auto_now=True)

    promoters = models.ForeignKey(
            'GenomicRegions',
            related_name='promoters',
            blank=True,
            null=True,
        )
    enhancers = models.ForeignKey(
            'GenomicRegions',
            related_name='enhancers',
            blank=True,
            null=True,
        )


class GenomicRegions(models.Model):
    name = models.CharField(
        max_length=32)
    assembly = models.ForeignKey('GenomeAssembly')
    bed_file = models.FileField()
    last_updated = models.DateTimeField(
        auto_now=True)


class Gene(models.Model):
    name = models.CharField(
        max_length=32)
    annotation = models.ForeignKey('GeneAnnotation')


class Transcript(models.Model):
    name = models.CharField(
        max_length=32)
    gene = models.ForeignKey('Gene')

    STRANDS = (('+', '+'), ('-', '-'))

    chromosome = models.CharField(max_length=32)
    strand = models.CharField(choices=STRANDS, max_length=1)
    start = models.IntegerField()
    end = models.IntegerField()
    exons = ArrayField(ArrayField(models.IntegerField(), size=2))
