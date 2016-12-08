import requests
import pyBigWig
import re
import numpy
import math

from django.db import models
from django.conf import settings
from django.contrib.postgres.fields import JSONField, ArrayField
from django.urls import reverse


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

    def get_user_counts(self):
        counts = dict()

        counts['favorites'] = len(UserFavorite.objects.filter(owner=self))
        counts['recommendations'] = len(UserRecommendation.objects.filter(owner=self))

        return counts

    def get_user_details(self, my_user=None):
        detail = dict()

        detail['username'] = self.user.username
        detail['pk'] = self.pk

        datasets = Dataset.objects.filter(owners__in=[self])
        detail['dataset_number'] = len(datasets)
        detail['data_favorited_by_number'] = 0
        for ds in datasets:
            detail['data_favorited_by_number'] += \
                len(DataFavorite.objects.filter(favorite=ds))

        detail['data_favorite_number'] = len(DataFavorite.objects.filter(owner=self))
        detail['user_favorite_number'] = len(UserFavorite.objects.filter(owner=self))
        detail['user_favorited_by_number'] = len(UserFavorite.objects.filter(favorite=self))

        if my_user:
            detail['is_favorite'] = self.is_favorite(my_user)
            detail['is_recommended'] = self.is_recommended(my_user)

        return detail

    def get_dataset_assembly_counts(self):
        datasets = Dataset.objects.filter(owners__in=[self])
        assembly_counts = dict()

        for ds in datasets:
            if ds.assembly.name in assembly_counts:
                assembly_counts[ds.assembly.name] += 1
            else:
                assembly_counts[ds.assembly.name] = 1

        return assembly_counts

    def get_dataset_experiment_type_counts(self):
        datasets = Dataset.objects.filter(owners__in=[self])
        experiment_type_counts = dict()

        for ds in datasets:
            if ds.data_type in experiment_type_counts:
                experiment_type_counts[ds.data_type] += 1
            else:
                experiment_type_counts[ds.data_type] = 1

        return experiment_type_counts

    def get_dataset_counts(self):
        favorite_dataset_counts = \
            len(DataFavorite.objects.filter(owner=self))
        personal_dataset_counts = \
            len(Dataset.objects.filter(owners__in=[self]))
        recommended_dataset_counts = \
            len(DataRecommendation.objects.filter(owner=self, hidden=False))

        return {
            'favorite_dataset_counts': favorite_dataset_counts,
            'personal_dataset_counts': personal_dataset_counts,
            'recommended_dataset_counts': recommended_dataset_counts,
        }

    def get_urls(self):
        add_favorite = reverse('api:user-add-favorite', kwargs={'pk': self.pk})
        remove_favorite = reverse('api:user-remove-favorite', kwargs={'pk': self.pk})
        hide_recommendation = reverse('api:user-hide-recommendation', kwargs={'pk': self.pk})
        detail = reverse('user', kwargs={'pk': self.pk})

        return {
            'add_favorite': add_favorite,
            'remove_favorite': remove_favorite,
            'hide_recommendation': hide_recommendation,
            'detail': detail,
        }

    def is_favorite(self, my_user):
        if UserFavorite.objects.filter(owner=my_user, favorite=self).exists():
            return 'true'
        else:
            return 'false'

    def is_recommended(self, my_user):
        if UserRecommendation.objects.filter(owner=my_user, recommended=self).exists():
            return 'true'
        else:
            return 'false'

    def get_display_data(self, my_user):
        plot_data = dict()
        plot_data['assembly_counts'] = self.get_dataset_assembly_counts()
        plot_data['experiment_counts'] = self.get_dataset_experiment_type_counts()

        meta_data = self.get_user_details(my_user)
        urls = self.get_urls()

        return {
            'plot_data': plot_data,
            'meta_data': meta_data,
            'urls': urls,
        }

    def get_personal_dataset_ids(self):
        datasets = []

        for ds in Dataset.objects.filter(owners__in=[self]):
            datasets.append({
                'id': ds.pk,
                'name': ds.name,
            })

        return datasets

    def get_favorite_dataset_ids(self):
        datasets = []

        for ds in [df.favorite for df in DataFavorite.objects.filter(owner=self)]:
            datasets.append({
                'id': ds.pk,
                'name': ds.name,
            })

        return datasets


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
    antibody = models.CharField(max_length=128, blank=True)

    description = models.TextField(blank=True)

    ambiguous_url = models.URLField()
    plus_url = models.URLField()
    minus_url = models.URLField()

    owners = models.ManyToManyField('MyUser', blank=True)
    name = models.CharField(max_length=128)
    slug = models.CharField(
        max_length=128)
    created = models.DateTimeField(
        auto_now_add=True)
    assembly = models.ForeignKey('GenomeAssembly')

    promoter_intersection = models.ForeignKey(
        'IntersectionValues', related_name='promoter', blank=True, null=True)
    enhancer_intersection = models.ForeignKey(
        'IntersectionValues', related_name='enhancer', blank=True, null=True)

    promoter_metaplot = models.ForeignKey(
        'MetaPlot', related_name='promoter', blank=True, null=True)
    enhancer_metaplot = models.ForeignKey(
        'MetaPlot', related_name='enhancer', blank=True, null=True)

    class Meta:
        get_latest_by = 'created'

    def get_absolute_url(self):
        return reverse('dataset', kwargs={'pk': self.pk})

    def get_favorites_count(self):
        pass

    def get_urls(self):
        add_favorite = reverse('api:dataset-add-favorite', kwargs={'pk': self.pk})
        remove_favorite = reverse('api:dataset-remove-favorite', kwargs={'pk': self.pk})
        hide_recommendation = reverse('api:dataset-hide-recommendation', kwargs={'pk': self.pk})

        edit = reverse('update_dataset', kwargs={'pk': self.pk})
        delete = reverse('delete_dataset', kwargs={'pk': self.pk})
        detail = reverse('dataset', kwargs={'pk': self.pk})

        return {
            'add_favorite': add_favorite,
            'remove_favorite': remove_favorite,
            'hide_recommendation': hide_recommendation,
            'edit': edit,
            'delete': delete,
            'detail': detail,
        }

    def is_favorite(self, my_user):
        if DataFavorite.objects.filter(owner=my_user, favorite=self).exists():
            return 'true'
        else:
            return 'false'

    def is_recommended(self, my_user):
        if DataRecommendation.objects.filter(owner=my_user, recommended=self).exists():
            return 'true'
        else:
            return 'false'

    def get_display_data(self, my_user):
        plot_data = dict()
        if self.promoter_metaplot:
            plot_data['promoter_metaplot'] = self.promoter_metaplot.meta_plot
        if self.enhancer_metaplot:
            plot_data['enhancer_metaplot'] = self.enhancer_metaplot.meta_plot
        meta_data = self.get_metadata(my_user)
        urls = self.get_urls()

        return {
            'plot_data': plot_data,
            'meta_data': meta_data,
            'urls': urls,
        }

    @staticmethod
    def check_valid_url(url):
        # ensure URL is valid and doesn't raise a 400/500 error
        try:
            resp = requests.head(url)
        except requests.exceptions.ConnectionError:
            return False, '{} not found.'.format(url)
        else:
            return resp.ok, '{}: {}'.format(resp.status_code, resp.reason)

    @staticmethod
    def get_browser_view(chromosome, start, end, datasets):

        # query_split = re.split(':|-|\s', query)
        # try:
        #     chromosome, start, end = query_split
        #     start = int(start) - 1
        #     end = int(end)
        # except:
        #     return 'Format error'
        # out_query = {
        #     'chromosome': chromosome,
        #     'start': start,
        #     'end': end,
        # }
        start = int(start) - 1
        end = int(end)

        data_ids = [int(d) for d in datasets.split(',')]
        out_data = []
        # out_dict = {
        #     'chromsome': chromosome,
        #     'start': start,
        #     'end': end,
        # }

        for _id in data_ids:
            ds = Dataset.objects.get(pk=_id)

            # if 'assembly' not in out_query:
            #     out_query['assembly'] = ds.assembly.name
            # elif ds.assembly.name != out_query['assembly']:
            #     raise ValueError('Assembly is not consistent across datasets.')

            if ds.ambiguous_url:
                bigwig = pyBigWig.open(ds.ambiguous_url)
                intervals = bigwig.intervals(chromosome, start, end)

                _range = end - start
                _interval = _range / 1000

                bins = []
                for i in range(1000):
                    bins.append([
                        math.ceil(start + _interval * i),
                        math.ceil(start + _interval * (i + 1)),
                        0,
                    ])

                interval_n = 0
                bin_n = 0

                while interval_n < len(intervals) and bin_n < len(bins):
                    if intervals[interval_n][0] < bins[bin_n][1] and \
                            bins[bin_n][0] <= intervals[interval_n][1]:
                        if intervals[interval_n][2] > bins[bin_n][2]:
                            bins[bin_n][2] = intervals[interval_n][2]
                        bin_n += 1
                    elif intervals[interval_n][1] < bins[bin_n][0]:
                        interval_n += 1
                    elif bins[bin_n][1] < intervals[interval_n][0] + 1:
                        bin_n += 1

                # out_dict['ambig_intervals'] = bins
                out_data.append({
                    'ambig_intervals': bins,
                    'id': _id,
                    'name': ds.name,
                    'assembly': ds.assembly.name,
                })

        # else:
        #     bigwig_1 = pyBigWig.open(self.plus_url)
        #     bigwig_2 = pyBigWig.open(self.minus_url)
        #     intervals_1 = bigwig_1.intervals(chromosome, start, end)
        #     intervals_2 = bigwig_2.intervals(chromosome, start, end)
        #
        #     out_dict['plus_intervals'] = intervals_1
        #     out_dict['minus_intervals'] = intervals_2

        return out_data

    def get_metadata(self, my_user=None):
        metadata = dict()

        metadata['id'] = self.id
        metadata['name'] = self.name
        metadata['data_type'] = self.data_type
        metadata['cell_type'] = self.cell_type

        if self.antibody:
            metadata['antibody'] = self.antibody
        if self.ambiguous_url:
            metadata['strand'] = 'Unstranded'
        else:
            metadata['strand'] = 'Stranded'
        if self.description:
            metadata['description'] = self.description

        metadata['owners'] = []
        if self.owners:
            for owner in self.owners.all():
                metadata['owners'].append(owner.user.username)

        if my_user:
            metadata['is_favorite'] = self.is_favorite(my_user)
            metadata['is_recommended'] = self.is_recommended(my_user)

        return metadata


class MetaPlot(models.Model):
    genomic_regions = models.ForeignKey('GenomicRegions')
    bigwig_url = models.URLField()

    relative_start = models.IntegerField()
    relative_end = models.IntegerField()
    meta_plot = JSONField()
    last_updated = models.DateTimeField(
        auto_now=True)


class IntersectionValues(models.Model):
    genomic_regions = models.ForeignKey('GenomicRegions')
    bigwig_url = models.URLField()

    relative_start = models.IntegerField()
    relative_end = models.IntegerField()
    intersection_values = JSONField()
    last_updated = models.DateTimeField(
        auto_now=True)


class Favorite(models.Model):
    owner = models.ForeignKey('MyUser')
    last_updated = models.DateTimeField(
        auto_now=True)

    class Meta:
        abstract = True


class UserFavorite(Favorite):
    favorite = models.ForeignKey('MyUser', related_name='favorite')


class DataFavorite(Favorite):
    favorite = models.ForeignKey('Dataset')


class Recommendation(models.Model):
    owner = models.ForeignKey('MyUser')
    last_updated = models.DateTimeField(
        auto_now=True)
    score = models.FloatField()
    hidden = models.BooleanField(default=False)

    class Meta:
        abstract = True
        ordering = ('score', '-last_updated',)


class UserRecommendation(Recommendation):
    recommended = models.ForeignKey('MyUser', related_name='recommended')


class DataRecommendation(Recommendation):
    recommended = models.ForeignKey('Dataset')
    reference_dataset = models.ForeignKey('Dataset', related_name='reference')

    def get_recommendation_data(self, my_user):
        plot_data = dict()
        plot_data['rec_promoter_intersection'] = \
            self.recommended.promoter_intersection.intersection_values
        plot_data['ref_promoter_intersection'] = \
            self.reference_dataset.promoter_intersection.intersection_values
        plot_data['rec_enhancer_intersection'] = \
            self.recommended.enhancer_intersection.intersection_values
        plot_data['ref_enhancer_intersection'] = \
            self.reference_dataset.enhancer_intersection.intersection_values
        meta_data = self.recommended.get_metadata(my_user)
        meta_data['reference_name'] = self.reference_dataset.name
        urls = self.recommended.get_urls()

        urls['reference_detail'] = \
            reverse('dataset', kwargs={'pk': self.reference_dataset.pk})

        return {
            'plot_data': plot_data,
            'meta_data': meta_data,
            'urls': urls,
        }


class CorrelationCell(models.Model):
    x_dataset = models.ForeignKey('Dataset', related_name='x')
    y_dataset = models.ForeignKey('Dataset', related_name='y')
    genomic_regions = models.ForeignKey('GenomicRegions')

    score = models.FloatField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (
            ('x_dataset', 'y_dataset', 'genomic_regions',),
        )

    @staticmethod
    def get_correlation_stats(regions):
        corr_values = []
        for corr in CorrelationCell.objects.filter(genomic_regions=regions):
            corr_values.append(corr.score)
        if corr_values:
            return (
                numpy.mean(corr_values),
                numpy.std(corr_values),
            )
        else:
            return (None, None)

    @staticmethod
    def get_z_score_list():
        z_scores = []

        corr_stats = dict()
        for assembly in GenomeAssembly.objects.all():
            promoter_mean, promoter_stdev = \
                CorrelationCell.get_correlation_stats(assembly.default_annotation.promoters)
            enhancer_mean, enhancer_stdev = \
                CorrelationCell.get_correlation_stats(assembly.default_annotation.enhancers)
            corr_stats[assembly] = {
                'promoter_mean': promoter_mean,
                'promoter_stdev': promoter_stdev,
                'enhancer_mean': enhancer_mean,
                'enhancer_stdev': enhancer_stdev,
            }

        for assembly in GenomeAssembly.objects.all():
            datasets = Dataset.objects.filter(assembly=assembly).order_by('id')
            for i, ds_1 in enumerate(datasets):
                for j, ds_2 in enumerate(datasets[i + 1:]):
                    scores = []
                    for corr in CorrelationCell.objects.filter(x_dataset=ds_1, y_dataset=ds_2):
                        if corr.genomic_regions == assembly.default_annotation.promoters:
                            mean = corr_stats[assembly]['promoter_mean']
                            stdev = corr_stats[assembly]['promoter_stdev']
                        elif corr.genomic_regions == assembly.default_annotation.enhancers:
                            mean = corr_stats[assembly]['enhancer_mean']
                            stdev = corr_stats[assembly]['enhancer_stdev']
                        scores.append((corr.score - mean) / stdev)
                    if scores:
                        z_scores.append({
                            'dataset_1': ds_1.id,
                            'dataset_2': ds_2.id,
                            'users_1': [user.id for user in ds_1.owners.all()],
                            'users_2': [user.id for user in ds_2.owners.all()],
                            'max_z_score': max(scores),
                        })

        return z_scores


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

    def __str__(self):
        return self.name


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
