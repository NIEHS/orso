import requests

from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import JSONField


class User(User):
    slug = models.CharField(
        max_length=128)

    favoriteUsers = models.ManyToManyField('User', symmetrical=False)
    favoriteData = models.ManyToManyField('Dataset')
    favoritesCount = models.IntegerField(default=0)
    dataFavoritesCount = models.IntegerField(default=0)


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

    owners = models.ManyToManyField('User')
    name = models.CharField(max_length=128)
    slug = models.CharField(
        max_length=128)
    favoritesCount = models.IntegerField(default=0)
    intersectionValues = models.FileField()
    created = models.DateTimeField(
        auto_now_add=True)

    promoterMetaPlot = models.ForeignKey('Meta', related_name='promoter_meta')
    enhancerMetaPlot = models.ForeignKey('Meta', related_name='enhancer_meta')

    @staticmethod
    def check_valid_url(url):
        # ensure URL is valid and doesn't raise a 400/500 error
        try:
            resp = requests.head(url)
        except requests.exceptions.ConnectionError:
            return False, '{} not found.'.format(url)
        else:
            return resp.ok, '{}: {}'.format(resp.status_code, resp.reason)


class Meta(models.Model):
    genomic_regions = models.ForeignKey('GenomicRegions')

    relative_start = models.IntegerField()
    relative_end = models.IntegerField()
    meta_plot = JSONField()


class Recommendations(models.Model):
    owner = models.ForeignKey('User')
    last_updated = models.DateTimeField(
        auto_now=True)


class UserRecommendations(Recommendations):
    rec_list = models.ManyToManyField('User')


class DataRecommendations(Recommendations):
    rec_list = models.ManyToManyField('Dataset')


class CorrelationMatrix(models.Model):
    matrix = JSONField()
    last_updated = models.DateTimeField(
        auto_now=True)


class GenomeAssembly(models.Model):
    name = models.CharField(
        unique=True,
        max_length=32)
    defaultAnnotation = models.ForeignKey('GeneAnnotation')


class GeneAnnotation(models.Model):
    assembly = models.ForeignKey('GenomeAssembly')
    gtf_file = models.FileField()

    promoters = models.ForeignKey('GenomicRegions', related_name='promoters')
    enhancers = models.ForeignKey('GenomicRegions', related_name='enhancers')


class GenomicRegions(models.Model):
    bed_file = models.FileField()


class Gene(models.Model):
    annotation = models.ForeignKey(GeneAnnotation)

    names = JSONField()

    STRANDS = (('+', '+'), ('-', '-'))

    chromosome = models.CharField(max_length=32)
    strand = models.CharField(choices=STRANDS, max_length=1)
    start = models.IntegerField()
    end = models.IntegerField()


class Exon(models.Model):
    gene = models.ForeignKey(Gene)
    id_num = models.IntegerField()

    start = models.IntegerField()
    end = models.IntegerField()
