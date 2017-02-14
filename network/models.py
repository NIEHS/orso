import requests
import pyBigWig
import numpy
import math
from collections import defaultdict

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
        counts['recommendations'] = \
            len(UserRecommendation.objects.filter(owner=self))

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
                len(ExperimentFavorite.objects.filter(favorite=ds))

        detail['data_favorite_number'] = \
            len(ExperimentFavorite.objects.filter(owner=self))
        detail['user_favorite_number'] = \
            len(UserFavorite.objects.filter(owner=self))
        detail['user_favorited_by_number'] = \
            len(UserFavorite.objects.filter(favorite=self))

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

    def get_experiment_counts(self):
        favorite_experiment_counts = \
            len(ExperimentFavorite.objects.filter(owner=self))
        personal_experiment_counts = \
            len(Experiment.objects.filter(owners__in=[self]))
        recommended_experiment_counts = \
            len(ExperimentRecommendation.objects.filter(owner=self, hidden=False))  # noqa

        return {
            'favorite_experiment_counts': favorite_experiment_counts,
            'personal_experiment_counts': personal_experiment_counts,
            'recommended_experiment_counts': recommended_experiment_counts,
        }

    def get_urls(self):
        add_favorite = reverse('api:user-add-favorite', kwargs={'pk': self.pk})
        remove_favorite = \
            reverse('api:user-remove-favorite', kwargs={'pk': self.pk})
        hide_recommendation = \
            reverse('api:user-hide-recommendation', kwargs={'pk': self.pk})
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
        if UserRecommendation.objects.filter(owner=my_user, recommended=self).exists():  # noqa
            return 'true'
        else:
            return 'false'

    def get_display_data(self, my_user):
        plot_data = dict()
        plot_data['assembly_counts'] = self.get_dataset_assembly_counts()
        plot_data['experiment_counts'] = \
            self.get_dataset_experiment_type_counts()

        meta_data = self.get_user_details(my_user)
        urls = self.get_urls()

        return {
            'plot_data': plot_data,
            'meta_data': meta_data,
            'urls': urls,
        }

    def get_personal_experiment_ids(self):
        experiments = []

        for exp in Experiment.objects.filter(owners__in=[self]):
            experiments.append({
                'id': exp.pk,
                'name': exp.name,
            })

        return experiments

    def get_favorite_experiment_ids(self):
        experiments = []

        for exp in [df.favorite for df in ExperimentFavorite.objects.filter(owner=self)]:  # noqa
            experiments.append({
                'id': exp.pk,
                'name': exp.name,
            })

        return experiments


class Project(models.Model):
    owners = models.ManyToManyField('MyUser', blank=True)
    name = models.CharField(max_length=128)
    description = models.TextField(blank=True)


class Experiment(models.Model):
    DATA_TYPES = (
        ('RAMPAGE', 'RAMPAGE'),
        ('RNA-seq', 'RNA-seq'),
        ('HiC', 'HiC'),
        ('RNA-PET', 'RNA-PET'),
        ('DNase-seq', 'DNase-seq'),
        ('siRNA knockdown followed by RNA-seq',
            'siRNA knockdown followed by RNA-seq'),
        ('eCLIP', 'eCLIP'),
        ('ChIA-PET', 'ChIA-PET'),
        ('shRNA knockdown followed by RNA-seq',
            'shRNA knockdown followed by RNA-seq'),
        ('single cell isolation followed by RNA-seq',
            'single cell isolation followed by RNA-seq'),
        ('Repli-chip', 'Repli-chip'),
        ('CRISPR genome editing followed by RNA-seq',
            'CRISPR genome editing followed by RNA-seq'),
        ('RIP-seq', 'RIP-seq'),
        ('whole-genome shotgun bisulfite sequencing',
            'whole-genome shotgun bisulfite sequencing'),
        ('ATAC-seq', 'ATAC-seq'),
        ('CAGE', 'CAGE'),
        ('MNase-seq', 'MNase-seq'),
        ('FAIRE-seq', 'FAIRE-seq'),
        ('ChIP-seq', 'ChIP-seq'),
        ('Repli-seq', 'Repli-seq'),
        ('microRNA-seq', 'microRNA-seq'),
    )

    data_type = models.CharField(
        max_length=64,
        choices=DATA_TYPES)
    cell_type = models.CharField(max_length=128)
    target = models.CharField(max_length=128, blank=True)
    owners = models.ManyToManyField('MyUser', blank=True)
    project = models.ForeignKey('Project', blank=True, null=True)
    name = models.CharField(max_length=128)
    slug = models.CharField(max_length=128)
    description = models.TextField(blank=True)

    class Meta:
        get_latest_by = 'created'

    def get_absolute_url(self):
        return reverse('experiment', kwargs={'pk': self.pk})

    def get_urls(self):
        add_favorite = \
            reverse('api:experiment-add-favorite', kwargs={'pk': self.pk})
        remove_favorite = \
            reverse('api:experiment-remove-favorite', kwargs={'pk': self.pk})
        hide_recommendation = \
            reverse('api:experiment-hide-recommendation', kwargs={'pk': self.pk})

        edit = reverse('update_experiment', kwargs={'pk': self.pk})
        delete = reverse('delete_experiment', kwargs={'pk': self.pk})
        detail = reverse('experiment', kwargs={'pk': self.pk})

        return {
            'add_favorite': add_favorite,
            'remove_favorite': remove_favorite,
            'hide_recommendation': hide_recommendation,
            'edit': edit,
            'delete': delete,
            'detail': detail,
        }

    def is_favorite(self, my_user):
        if ExperimentFavorite.objects.filter(owner=my_user, favorite=self).exists():  # noqa
            return 'true'
        else:
            return 'false'

    def is_recommended(self, my_user):
        if ExperimentRecommendation.objects.filter(owner=my_user, recommended=self).exists():  # noqa
            return 'true'
        else:
            return 'false'

    def get_average_metaplots(self, assemblies=None):
        if assemblies:
            datasets = []
            for assembly in assemblies:
                assembly_obj = GenomeAssembly.objects.get(pk=assembly)
                datasets.extend(Dataset.objects.filter(
                    experiment=self, assembly=assembly_obj))
        else:
            datasets = Dataset.objects.filter(experiment=self)

        average_metaplots = dict()
        assembly_count = defaultdict(int)

        for ds in datasets:
            assembly_count[ds.assembly.name] += 1
            if ds.assembly.name in average_metaplots:
                for i, entry in enumerate(
                    ds.promoter_metaplot.meta_plot['metaplot_values']
                ):
                    average_metaplots[ds.assembly.name]['promoters'][i] += \
                        entry
                for i, entry in enumerate(
                    ds.enhancer_metaplot.meta_plot['metaplot_values']
                ):
                    average_metaplots[ds.assembly.name]['enhancers'][i] += \
                        entry
            else:
                average_metaplots[ds.assembly.name] = {
                    'promoters': ds.promoter_metaplot.meta_plot,
                    'enhancers': ds.enhancer_metaplot.meta_plot,
                }

        #  Divide by assembly counts
        for assembly in average_metaplots.keys():
            count = assembly_count[assembly]
            for i, entry in enumerate(
                average_metaplots[assembly]['promoters']['metaplot_values']
            ):
                average_metaplots[assembly]['promoters']['metaplot_values'][i] = entry / count  # noqa
            for i, entry in enumerate(
                average_metaplots[assembly]['enhancers']['metaplot_values']
            ):
                average_metaplots[assembly]['enhancers']['metaplot_values'][i] = entry / count  # noqa

        return average_metaplots

    def get_average_intersections(self, assemblies=None):
        if assemblies:
            datasets = []
            for assembly in assemblies:
                assembly_obj = GenomeAssembly.objects.get(pk=assembly)
                datasets.extend(Dataset.objects.filter(
                    experiment=self, assembly=assembly_obj))
        else:
            datasets = Dataset.objects.filter(experiment=self)

        average_intersections = dict()
        assembly_count = defaultdict(int)

        for ds in datasets:
            assembly_count[ds.assembly.name] += 1
            if ds.assembly.name in average_intersections:
                for i, entry in enumerate(
                    ds.promoter_intersection.intersection_values
                ):
                    average_intersections[ds.assembly.name]['promoters'][i] += entry  # noqa
                for i, entry in enumerate(
                    ds.enhancer_intersection.intersection_values
                ):
                    average_intersections[ds.assembly.name]['enhancers'][i] += entry  # noqa
            else:
                average_intersections[ds.assembly.name] = {
                    'promoters': ds.promoter_intersection.intersection_values,
                    'enhancers': ds.enhancer_intersection.intersection_values,
                }

        #  Divide by assembly counts
        for assembly in average_intersections.keys():
            count = assembly_count[assembly]
            for i, entry in enumerate(
                average_intersections[assembly]['promoters']
            ):
                average_intersections[assembly]['promoters'][i] = entry / count
            for i, entry in enumerate(
                average_intersections[assembly]['enhancers']
            ):
                average_intersections[assembly]['enhancers'][i] = entry / count

        return average_intersections

    def get_display_data(self, my_user):
        return {
            'plot_data': self.get_average_metaplots(),
            'meta_data': self.get_metadata(my_user),
            'urls': self.get_urls(),
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
        # TODO: iterate through datasets to get browser data
        start = int(start) - 1
        end = int(end)

        data_ids = [int(d) for d in datasets.split(',')]
        out_data = []

        for _id in data_ids:
            ds = Dataset.objects.get(pk=_id)

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

                out_data.append({
                    'ambig_intervals': bins,
                    'id': _id,
                    'name': ds.name,
                    'assembly': ds.assembly.name,
                })

        return out_data

    def get_metadata(self, my_user=None):
        metadata = dict()

        metadata['id'] = self.id
        metadata['name'] = self.name
        metadata['data_type'] = self.data_type
        metadata['cell_type'] = self.cell_type

        if self.target:
            metadata['target'] = self.target
        # if self.ambiguous_url:
        #     metadata['strand'] = 'Unstranded'
        # else:
        #     metadata['strand'] = 'Stranded'
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


class Dataset(models.Model):

    description = models.TextField(blank=True)

    ambiguous_url = models.URLField()
    plus_url = models.URLField()
    minus_url = models.URLField()

    experiment = models.ForeignKey('Experiment', blank=True, null=True)
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

    # def get_urls(self):
    #     add_favorite = \
    #         reverse('api:dataset-add-favorite', kwargs={'pk': self.pk})
    #     remove_favorite = \
    #         reverse('api:dataset-remove-favorite', kwargs={'pk': self.pk})
    #     hide_recommendation = \
    #         reverse('api:dataset-hide-recommendation', kwargs={'pk': self.pk})
    #
    #     edit = reverse('update_dataset', kwargs={'pk': self.pk})
    #     delete = reverse('delete_dataset', kwargs={'pk': self.pk})
    #     detail = reverse('dataset', kwargs={'pk': self.pk})
    #
    #     return {
    #         'add_favorite': add_favorite,
    #         'remove_favorite': remove_favorite,
    #         'hide_recommendation': hide_recommendation,
    #         'edit': edit,
    #         'delete': delete,
    #         'detail': detail,
    #     }
    #
    # def is_favorite(self, my_user):
    #     if DataFavorite.objects.filter(owner=my_user, favorite=self).exists():
    #         return 'true'
    #     else:
    #         return 'false'
    #
    # def is_recommended(self, my_user):
    #     if DataRecommendation.objects.filter(owner=my_user, recommended=self).exists():  # noqa
    #         return 'true'
    #     else:
    #         return 'false'

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

        start = int(start) - 1
        end = int(end)

        data_ids = [int(d) for d in datasets.split(',')]
        out_data = []

        for _id in data_ids:
            ds = Dataset.objects.get(pk=_id)

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

                out_data.append({
                    'ambig_intervals': bins,
                    'id': _id,
                    'name': ds.name,
                    'assembly': ds.assembly.name,
                })

        return out_data

    # def get_metadata(self, my_user=None):
    #     metadata = dict()
    #
    #     metadata['id'] = self.id
    #     metadata['name'] = self.name
    #     metadata['data_type'] = self.data_type
    #     metadata['cell_type'] = self.cell_type
    #
    #     if self.target:
    #         metadata['target'] = self.target
    #     if self.ambiguous_url:
    #         metadata['strand'] = 'Unstranded'
    #     else:
    #         metadata['strand'] = 'Stranded'
    #     if self.description:
    #         metadata['description'] = self.description
    #
    #     metadata['owners'] = []
    #     if self.owners:
    #         for owner in self.owners.all():
    #             metadata['owners'].append(owner.user.username)
    #
    #     if my_user:
    #         metadata['is_favorite'] = self.is_favorite(my_user)
    #         metadata['is_recommended'] = self.is_recommended(my_user)
    #
    #     return metadata


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


class ExperimentFavorite(Favorite):
    favorite = models.ForeignKey('Experiment')


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


class ExperimentRecommendation(Recommendation):
    recommended = models.ForeignKey('Experiment')
    reference_experiment = models.ForeignKey('Experiment', related_name='reference')  # noqa

    #  TODO: fix for Experiment model
    def get_recommendation_data(self, my_user):
        plot_data = dict()

        rec_assemblies = set()
        ref_assemblies = set()
        for ds in Dataset.objects.filter(experiment=self.recommended):
            rec_assemblies.add(ds.assembly.id)
        for ds in Dataset.objects.filter(experiment=self.reference_experiment):
            ref_assemblies.add(ds.assembly.id)
        shared_assemblies = rec_assemblies & ref_assemblies

        # for assembly_id in shared_assemblies:
        #     assembly = GenomeAssembly.objects.get(pk=assembly_id)
        #     plot_data[assembly.name] = dict()
        plot_data = dict()
        plot_data['rec'] = self.recommended.get_average_intersections(
            assemblies=shared_assemblies)
        plot_data['ref'] = self.reference_experiment.get_average_intersections(
            assemblies=shared_assemblies)
    #     plot_data['rec_promoter_intersection'] = \
    #         self.recommended.promoter_intersection.intersection_values
    #     plot_data['ref_promoter_intersection'] = \
    #         self.reference_experiment.promoter_intersection.intersection_values
    #     plot_data['rec_enhancer_intersection'] = \
    #         self.recommended.enhancer_intersection.intersection_values
    #     plot_data['ref_enhancer_intersection'] = \
    #         self.reference_experiment.enhancer_intersection.intersection_values
        meta_data = self.recommended.get_metadata(my_user)
        meta_data['reference_name'] = self.reference_experiment.name
        urls = self.recommended.get_urls()
        urls['reference_detail'] = \
            reverse('experiment', kwargs={'pk': self.reference_experiment.pk})
    #
        return {
            'plot_data': plot_data,
            'meta_data': meta_data,
            'urls': urls,
        }


class CorrelationCell(models.Model):
    x_experiment = models.ForeignKey('Experiment', related_name='x')
    y_experiment = models.ForeignKey('Experiment', related_name='y')
    genomic_regions = models.ForeignKey('GenomicRegions')

    score = models.FloatField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (
            ('x_experiment', 'y_experiment', 'genomic_regions',),
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

    #  TODO: check for new Project>Experiment>Dataset hierarchy
    @staticmethod
    def get_z_score_list():
        z_scores = []

        corr_stats = dict()
        for assembly in GenomeAssembly.objects.all():
            promoter_mean, promoter_stdev = \
                CorrelationCell.get_correlation_stats(assembly.default_annotation.promoters)  # noqa
            enhancer_mean, enhancer_stdev = \
                CorrelationCell.get_correlation_stats(assembly.default_annotation.enhancers)  # noqa
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
                    for corr in CorrelationCell.objects.filter(
                            x_experiment=ds_1,
                            y_experiment=ds_2):
                        if corr.genomic_regions == \
                                assembly.default_annotation.promoters:
                            mean = corr_stats[assembly]['promoter_mean']
                            stdev = corr_stats[assembly]['promoter_stdev']
                        elif corr.genomic_regions == \
                                assembly.default_annotation.enhancers:
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
