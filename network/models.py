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

        experiments = Experiment.objects.filter(owners__in=[self])
        detail['dataset_number'] = len(experiments)
        detail['data_favorited_by_number'] = 0
        for exp in experiments:
            detail['data_favorited_by_number'] += \
                len(ExperimentFavorite.objects.filter(favorite=exp))

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
        experiments = Experiment.objects.filter(owners__in=[self])
        datasets = []
        for exp in experiments:
            datasets.extend(Dataset.objects.filter(experiment=exp))
        assembly_counts = dict()

        for ds in datasets:
            if ds.assembly.name in assembly_counts:
                assembly_counts[ds.assembly.name] += 1
            else:
                assembly_counts[ds.assembly.name] = 1

        return assembly_counts

    def get_dataset_experiment_type_counts(self):
        datasets = Experiment.objects.filter(owners__in=[self])
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
    created = models.DateTimeField(auto_now_add=True)

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

    def get_assemblies(self):
        assemblies = set()
        for ds in Dataset.objects.filter(experiment=self):
            assemblies.add(ds.assembly)
        return assemblies

    def get_average_metaplots(self):
        average_metaplots = dict()
        counts = defaultdict(int)

        #  Add similar metaplots together
        for mp in MetaPlot.objects.filter(dataset__experiment=self):
            gr = mp.genomic_regions
            counts[gr] += 1
            if gr in average_metaplots:
                for i, entry in enumerate(mp.meta_plot['metaplot_values']):
                    average_metaplots[gr]['metaplot_values'][i] += \
                        entry
            else:
                average_metaplots[gr] = mp.meta_plot

        #  Divide by assembly counts
        for gr in average_metaplots.keys():
            count = counts[gr]
            for i, entry in enumerate(
                average_metaplots[gr]['metaplot_values']
            ):
                average_metaplots[gr]['metaplot_values'][i] = entry / count

        #  Put into output format
        out = []
        for gr, metaplot in average_metaplots.items():
            out.append({
                'regions': gr.short_label,
                'regions_pk': gr.pk,
                'assembly': gr.assembly.name,
                'metaplot': metaplot,
            })

        return out

    def get_average_intersections(self):
        average_intersections = dict()
        counts = defaultdict(int)

        #  Add similar intersections together
        for iv in IntersectionValues.objects.filter(dataset__experiment=self):
            gr = iv.genomic_regions
            counts[gr] += 1
            if gr in average_intersections:
                for i, entry in enumerate(iv.intersection_values):
                    average_intersections[gr][i] += entry
            else:
                average_intersections[gr] = iv.intersection_values

        #  Divide by assembly counts
        for gr in average_intersections.keys():
            count = counts[gr]
            for i, entry in enumerate(average_intersections[gr]):
                average_intersections[gr][i] = entry / count

        #  Put into output format
        out = []
        for gr, intersection_values in average_intersections.items():
            out.append({
                'regions': gr.short_label,
                'regions_pk': gr.pk,
                'assembly': gr.assembly.name,
                'intersection_values': intersection_values,
            })

        return out

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
        if self.description:
            metadata['description'] = self.description

        metadata['owners'] = []
        if self.owners:
            for owner in self.owners.all():
                metadata['owners'].append(owner.user.username)

        metadata['assemblies'] = []
        for assembly in self.get_assemblies():
            metadata['assemblies'].append(assembly.name)

        if my_user:
            metadata['is_favorite'] = self.is_favorite(my_user)
            metadata['is_recommended'] = self.is_recommended(my_user)

        return metadata


class Dataset(models.Model):
    #  TODO: change promoter/enhancer intersection to single intersection list
    #  TODO: change promoter/enhancer metaplots to single metaplot list
    description = models.TextField(blank=True)

    ambiguous_url = models.URLField(null=True, blank=True)
    plus_url = models.URLField(null=True, blank=True)
    minus_url = models.URLField(null=True, blank=True)

    experiment = models.ForeignKey('Experiment', blank=True, null=True)
    name = models.CharField(max_length=128)
    slug = models.CharField(
        max_length=128)
    created = models.DateTimeField(
        auto_now_add=True)
    assembly = models.ForeignKey('GenomeAssembly')

    class Meta:
        get_latest_by = 'created'

    def get_absolute_url(self):
        return reverse('dataset', kwargs={'pk': self.pk})

    def get_favorites_count(self):
        pass

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

    def is_stranded(self):
        return bool(self.plus_url) and bool(self.minus_url)


class MetaPlot(models.Model):
    genomic_regions = models.ForeignKey('GenomicRegions')
    dataset = models.ForeignKey('Dataset')
    meta_plot = JSONField()
    last_updated = models.DateTimeField(
        auto_now=True)


class IntersectionValues(models.Model):
    genomic_regions = models.ForeignKey('GenomicRegions')
    dataset = models.ForeignKey('Dataset')
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
    hidden = models.BooleanField(default=False)

    class Meta:
        abstract = True


class UserRecommendation(Recommendation):
    recommended = models.ForeignKey('MyUser', related_name='recommended')


class ExperimentRecommendation(Recommendation):
    recommended = models.ForeignKey('Experiment')

    correlation_rank = models.IntegerField()
    correlation_experiment = \
        models.ForeignKey('Experiment', related_name='correlation')

    metadata_rank = models.IntegerField()
    metadata_experiment = \
        models.ForeignKey('Experiment', related_name='metadata')

    collaborative_rank = models.IntegerField()

    class Meta:
        unique_together = (
            ('owner', 'recommended',),
        )

    def get_recommendation_data(self, my_user):
        plot_data = dict()

        rec_assemblies = set()
        ref_assemblies = set()
        for ds in Dataset.objects.filter(experiment=self.recommended):
            rec_assemblies.add(ds.assembly.id)
        for ds in Dataset.objects.filter(experiment=self.reference_experiment):
            ref_assemblies.add(ds.assembly.id)
        shared_assemblies = rec_assemblies & ref_assemblies

        plot_data = dict()
        plot_data['rec'] = self.recommended.get_average_intersections(
            assemblies=shared_assemblies)
        plot_data['ref'] = self.reference_experiment.get_average_intersections(
            assemblies=shared_assemblies)

        meta_data = self.recommended.get_metadata(my_user)
        meta_data['reference_name'] = self.reference_experiment.name
        urls = self.recommended.get_urls()
        urls['reference_detail'] = \
            reverse('experiment', kwargs={'pk': self.reference_experiment.pk})

        return {
            'plot_data': plot_data,
            'meta_data': meta_data,
            'urls': urls,
        }


class MetadataCorrelation(models.Model):
    x_experiment = models.ForeignKey('Experiment', related_name='x_meta')
    y_experiment = models.ForeignKey('Experiment', related_name='y_meta')

    score = models.FloatField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (
            ('x_experiment', 'y_experiment',),
        )

    @staticmethod
    def get_score(exp_1, exp_2):
        if exp_1.id < exp_2.id:
            return MetadataCorrelation.objects.get(
                x_experiment=exp_1, y_experiment=exp_2).score
        else:
            return MetadataCorrelation.objects.get(
                x_experiment=exp_2, y_experiment=exp_1).score


class ExperimentCorrelation(models.Model):
    x_experiment = models.ForeignKey('Experiment', related_name='x_value')
    y_experiment = models.ForeignKey('Experiment', related_name='y_value')
    genomic_regions = models.ForeignKey('GenomicRegions')

    score = models.FloatField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (
            ('x_experiment', 'y_experiment', 'genomic_regions',),
        )

    @staticmethod
    def get_score(exp_1, exp_2):
        scores = []
        if exp_1.id < exp_2.id:
            corrs = ExperimentCorrelation.objects.filter(
                x_experiment=exp_1, y_experiment=exp_2)
        else:
            corrs = ExperimentCorrelation.objects.filter(
                x_experiment=exp_2, y_experiment=exp_1)
        for c in corrs:
            scores.append(c.score)
        return max(scores)

    @staticmethod
    def get_correlation_stats(regions):
        corr_values = []
        for corr in ExperimentCorrelation.objects.filter(
                genomic_regions=regions):
            corr_values.append(corr.score)
        if corr_values:
            return (
                numpy.mean(corr_values),
                numpy.std(corr_values),
            )
        else:
            return (None, None)

    @staticmethod
    def get_max_z_scores():
        max_z_scores = defaultdict(float)
        z_scores = defaultdict(list)

        for gr in GenomicRegions.objects.all():
            mean, std_dev = ExperimentCorrelation.get_correlation_stats(gr)
            for corr in ExperimentCorrelation.objects.filter(
                    genomic_regions=gr):
                z_scores[(corr.x_experiment, corr.y_experiment)].append(
                    (corr.score - mean) / std_dev)

        for exps, scores in z_scores.items():
            max_z_scores[exps] = max(scores)

        return max_z_scores


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


class GenomicRegions(models.Model):
    name = models.CharField(
        max_length=32)
    assembly = models.ForeignKey('GenomeAssembly')
    bed_file = models.FileField()
    short_label = models.CharField(max_length=32)
    last_updated = models.DateTimeField(
        auto_now=True)

    variance = JSONField(blank=True, null=True)
    # Use to only display/consider regions with high variance across datasets
    variance_mask = JSONField(blank=True, null=True)

    pca = JSONField(blank=True, null=True)


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

    @staticmethod
    def get_transcripts_in_range(annotation, chromosome, start, end):
        return Transcript.objects.filter(
            gene__annotation=annotation,
            chromosome=chromosome,
            start__lte=end,
            end__gte=start,
        )


class TranscriptIntersection(models.Model):
    transcript = models.ForeignKey('Transcript')
    dataset = models.ForeignKey('Dataset')

    promoter_value = models.FloatField()
    genebody_value = models.FloatField()
    exon_values = ArrayField(models.FloatField())
    intron_values = ArrayField(models.FloatField())
