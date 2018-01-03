import requests
import pyBigWig
import numpy
import math
import json
import string
from collections import defaultdict

from django.db import models
from django.conf import settings
from django.contrib.postgres.fields import JSONField, ArrayField
from django.urls import reverse
from nltk.corpus import stopwords
from picklefield.fields import PickledObjectField
from scipy.stats import variation as coeff_variance

from analysis.ontology import Ontology as OntologyObject

STRANDS = (('+', '+'), ('-', '-'))
LOCUS_GROUP_TYPES = (
    ('promoter', 'promoter'),
    ('genebody', 'genebody'),
    ('mRNA', 'mRNA'),
    ('enhancer', 'enhancer'),
)
ONTOLOGY_TYPES = (
    ('GeneOntology', 'GeneOntology'),
    ('DiseaseOntology', 'DiseaseOntology'),
    ('CellOntology', 'CellOntology'),
    ('CellLineOntology', 'CellLineOntology'),
)


class MyUser(models.Model):
    favorite_users = models.ManyToManyField(
        'MyUser', symmetrical=False, blank=True)
    favorite_data = models.ManyToManyField('Dataset', blank=True)

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL)
    slug = models.CharField(
        max_length=128)

    class Meta:
        verbose_name = 'MyUser'
        verbose_name_plural = 'MyUsers'

    def get_user_favorites_count(self):
        pass

    def get_data_favorites_count(self):
        pass

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

    def get_experiment_type_counts(self):
        experiments = Experiment.objects.filter(owners__in=[self])
        experiment_type_counts = dict()

        for exp in experiments:
            if exp.experiment_type.name in experiment_type_counts:
                experiment_type_counts[exp.experiment_type.name] += 1
            else:
                experiment_type_counts[exp.experiment_type.name] = 1

        return experiment_type_counts

    def get_experiment_counts(self):
        favorite_experiment_counts = \
            len(ExperimentFavorite.objects.filter(owner=self))
        personal_experiment_counts = \
            len(Experiment.objects.filter(owners__in=[self]))

        return {
            'favorite_experiment_counts': favorite_experiment_counts,
            'personal_experiment_counts': personal_experiment_counts,
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

    def get_display_data(self, my_user):
        plot_data = dict()
        plot_data['assembly_counts'] = self.get_dataset_assembly_counts()
        plot_data['experiment_counts'] = self.get_experiment_type_counts()

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

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)


class ExperimentType(models.Model):
    '''
    Experiment type, i.e. sequencing technology. Largely used organizationally
    '''
    name = models.CharField(max_length=64)
    short_name = models.CharField(max_length=64)
    relevant_regions = models.CharField(
        choices=LOCUS_GROUP_TYPES, max_length=64)

    def __str__(self):
        return self.name


class Experiment(models.Model):
    experiment_type = models.ForeignKey('ExperimentType')
    owners = models.ManyToManyField('MyUser', blank=True)
    project = models.ForeignKey('Project', blank=True, null=True)

    cell_type = models.CharField(max_length=128)
    target = models.CharField(max_length=128, blank=True)
    name = models.CharField(max_length=128)
    slug = models.CharField(max_length=128)
    description = models.TextField(blank=True)
    created = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)

    consortial_id = models.CharField(max_length=128, null=True, default=None)
    processed = models.BooleanField(default=False)

    class Meta:
        get_latest_by = 'created'
        unique_together = ('project', 'consortial_id')

    def get_absolute_url(self):
        return reverse('experiment', kwargs={'pk': self.pk})

    def get_urls(self):
        add_favorite = reverse(
            'api:experiment-add-favorite', kwargs={'pk': self.pk})
        remove_favorite = reverse(
            'api:experiment-remove-favorite', kwargs={'pk': self.pk})
        hide_recommendation = reverse(
            'api:experiment-hide-recommendation', kwargs={'pk': self.pk})

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

    def get_assemblies(self):
        assemblies = set()
        for ds in Dataset.objects.filter(experiment=self):
            assemblies.add(ds.assembly)
        return assemblies

    def get_average_metaplots(self):
        average_metaplots = dict()
        counts = defaultdict(int)

        target_group_types = [
            'promoter',
            'genebody',
            'enhancer',
        ]

        #  Add similar metaplots together
        for mp in MetaPlot.objects.filter(
            dataset__experiment=self,
            locus_group__group_type__in=target_group_types,
        ):
            locus_group = mp.locus_group
            counts[locus_group] += 1
            metaplot = json.loads(mp.metaplot)
            if locus_group in average_metaplots:
                for i, entry in enumerate(metaplot['metaplot_values']):
                    average_metaplots[locus_group]['metaplot_values'][i] += \
                        entry
            else:
                average_metaplots[locus_group] = metaplot

        #  Divide by assembly counts
        for locus_group in average_metaplots.keys():
            count = counts[locus_group]
            for i, entry in enumerate(
                average_metaplots[locus_group]['metaplot_values']
            ):
                average_metaplots[locus_group]['metaplot_values'][i] = \
                    entry / count

        #  Put into output format
        out = []
        for locus_group, metaplot in average_metaplots.items():
            out.append({
                'regions': locus_group.group_type,
                'regions_pk': locus_group.pk,
                'assembly': locus_group.assembly.name,
                'metaplot': metaplot,
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
        metadata['data_type'] = self.experiment_type.name
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

        return metadata

    def get_word_set(self, ngram_level=3):
        '''
        Get set of words from Experiment metadata.
        '''
        def find_ngrams(input_list, n):
            return [' '.join(words) for words in
                    zip(*[input_list[i:] for i in range(n)])]

        # Make translation for removing punctuation
        trans = str.maketrans('', '', string.punctuation)

        word_source = '\n'.join([
            self.description,
            self.cell_type,
            self.target,
        ])

        word_source = word_source.translate(trans).lower()

        word_list = []

        for line in word_source.split('\n'):
            words = line.split()
            word_list.extend(words)
            for i in range(2, ngram_level + 1):
                word_list.extend(find_ngrams(words, i))

        # Remove NLTK stopwords
        word_list = \
            [x for x in word_list if x not in stopwords.words('english')]

        # Convert to set
        return set(word_list)


class Dataset(models.Model):
    assembly = models.ForeignKey('Assembly')
    experiment = models.ForeignKey('Experiment', blank=True, null=True)

    name = models.CharField(max_length=128)
    slug = models.CharField(max_length=128)
    created = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)
    description = models.TextField(blank=True)

    ambiguous_url = models.URLField(null=True, blank=True)
    plus_url = models.URLField(null=True, blank=True)
    minus_url = models.URLField(null=True, blank=True)

    consortial_id = models.CharField(max_length=128, null=True, default=None)
    processed = models.BooleanField(default=False)

    class Meta:
        get_latest_by = 'created'
        unique_together = ('experiment', 'consortial_id')

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

    def get_metaplots(self):
        target_group_types = [
            'promoter',
            'genebody',
            'enhancer',
        ]

        out = []
        for mp in MetaPlot.objects.filter(
            dataset=self,
            locus_group__group_type__in=target_group_types,
        ):
            out.append({
                'regions': mp.locus_group.group_type,
                'regions_pk': mp.locus_group.pk,
                'assembly': mp.locus_group.assembly.name,
                'metaplot': json.loads(mp.metaplot),
            })

        return out


class Favorite(models.Model):
    owner = models.ForeignKey('MyUser')

    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class UserFavorite(Favorite):
    favorite = models.ForeignKey('MyUser', related_name='favorite')


class ExperimentFavorite(Favorite):
    favorite = models.ForeignKey('Experiment')


class Assembly(models.Model):
    name = models.CharField(unique=True, max_length=32)
    chromosome_sizes = JSONField(blank=True, null=True)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    def get_transcripts(self):
        return Transcript.objects.filter(
            gene__annotation=self.annotation)

    def read_in_chrom_sizes(self, chrom_sizes_path):
        chrom_sizes = dict()
        with open(chrom_sizes_path) as f:
            for line in f:
                chromosome, size = line.strip().split()
                chrom_sizes[chromosome] = int(size)
        self.chromosome_sizes = json.dumps(chrom_sizes)
        self.save()


class Annotation(models.Model):
    assembly = models.ForeignKey('Assembly')

    name = models.CharField(max_length=32)
    last_updated = models.DateTimeField(auto_now=True)


class PCA(models.Model):
    '''
    scikit-learn PCA object. Used to transform locus intersections and
    find pairwise distances between datasets.
    '''
    locus_group = models.ForeignKey('LocusGroup')
    selected_loci = models.ManyToManyField(
        'Locus', through='PCALocusOrder')
    experiment_type = models.ForeignKey('ExperimentType')
    transformed_datasets = models.ManyToManyField(
        'Dataset', through='PCATransformedValues')

    plot = JSONField()
    pca = PickledObjectField()
    covariation_matrix = PickledObjectField(null=True, blank=True)
    inverse_covariation_matrix = PickledObjectField(null=True, blank=True)

    last_updated = models.DateTimeField(auto_now=True, null=True)


class PCATransformedValues(models.Model):
    '''
    Used to store PCA-transformed dataset values.
    '''
    pca = models.ForeignKey('PCA')
    dataset = models.ForeignKey('Dataset')

    transformed_values = ArrayField(models.FloatField(), size=3)
    last_updated = models.DateTimeField(auto_now=True)


class PCALocusOrder(models.Model):
    '''
    Used to store order for selected loci. Essential for transformations.
    '''
    pca = models.ForeignKey('PCA')
    locus = models.ForeignKey('Locus')

    order = models.PositiveIntegerField()
    last_updated = models.DateTimeField(auto_now=True)


class IDF(models.Model):
    '''
    JSON containing IDF values.
    '''
    assembly = models.ForeignKey('Assembly')
    experiment_type = models.ForeignKey('ExperimentType')

    idf = JSONField()
    last_updated = models.DateTimeField(auto_now=True)


class TfidfVectorizer(models.Model):
    '''
    scikit-learn TfidfVectorizer object.
    '''
    assembly = models.ForeignKey('Assembly')
    experiment_type = models.ForeignKey('ExperimentType')

    tfidf_vectorizer = PickledObjectField()
    last_updated = models.DateTimeField(auto_now=True)


class Locus(models.Model):
    group = models.ForeignKey('LocusGroup')
    enhancer = models.ForeignKey('Enhancer', blank=True, null=True)
    transcript = models.ForeignKey('Transcript', blank=True, null=True)

    strand = models.CharField(choices=STRANDS, max_length=1, null=True)
    chromosome = models.CharField(max_length=32)
    regions = ArrayField(ArrayField(models.IntegerField(), size=2))


class LocusGroup(models.Model):
    assembly = models.ForeignKey('Assembly')
    group_type = models.CharField(choices=LOCUS_GROUP_TYPES, max_length=32)


class Gene(models.Model):
    annotation = models.ForeignKey('Annotation')
    selected_transcript = models.ForeignKey(
        'Transcript', related_name='selecting', blank=True, null=True)

    name = models.CharField(
        max_length=32)

    def get_transcript_with_highest_expression(self):
        expression_values = dict()
        for transcript in Transcript.objects.filter(gene=self):
            expression_values[transcript] = []
            intersections = \
                DatasetIntersection.objects.filter(
                    locus__group__group_type='mRNA',
                    locus__transcript=transcript,
                    dataset__experiment__experiment_type__name='RNA-seq',
                )
            for intersection in intersections:
                expression_values[transcript].append(
                    intersection.normalized_value)
        if expression_values:
            return sorted(
                expression_values.items(),
                key=lambda x: (-numpy.median(x[1]), x[0].name),
            )[0][0]
        else:
            return None


class Transcript(models.Model):
    gene = models.ForeignKey('Gene')

    name = models.CharField(max_length=32)
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

    def get_intersection_variance(self, experiment_type_name,
                                  locus_group_type):
        '''
        For the transcript, return variance (coefficient of variance) across
        intersection values.
        '''
        values = DatasetIntersection.objects.filter(
            locus__transcript=self,
            locus__group__group_type=locus_group_type,
            dataset__experiment__experiment_type__name=experiment_type_name,
        ).values_list('normalized_value', flat=True)
        if numpy.mean(values) == 0:
            return 0.0
        else:
            return coeff_variance(values)

    def get_intersection_median(self, experiment_type_name,
                                locus_group_type):
        '''
        For the transcript, return the median expression values from
        intersections.
        '''
        values = DatasetIntersection.objects.filter(
            locus__transcript=self,
            locus__group__group_type=locus_group_type,
            dataset__experiment__experiment_type__name=experiment_type_name,
        ).values_list('normalized_value', flat=True)
        return numpy.median(values)


class Enhancer(models.Model):
    annotation = models.ForeignKey('Annotation')

    name = models.CharField(max_length=32)
    chromosome = models.CharField(max_length=32)
    start = models.IntegerField()
    end = models.IntegerField()


class DatasetIntersection(models.Model):
    locus = models.ForeignKey('Locus')
    dataset = models.ForeignKey('Dataset')

    raw_value = models.FloatField()
    normalized_value = models.FloatField()

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        unique_together = ('locus', 'dataset')


class DatasetIntersectionJson(models.Model):
    locus_group = models.ForeignKey('LocusGroup')
    dataset = models.ForeignKey('Dataset')

    intersection_values = JSONField()

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        unique_together = ('locus_group', 'dataset')


class ExperimentIntersection(models.Model):
    locus = models.ForeignKey('Locus')
    experiment = models.ForeignKey('Experiment')

    average_value = models.FloatField()

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        unique_together = ('locus', 'experiment')


class DatasetDistance(models.Model):
    '''
    Base model for dataset distances.
    '''
    dataset_1 = models.ForeignKey(
        'Dataset', related_name='%(app_label)s_%(class)s_first')
    dataset_2 = models.ForeignKey(
        'Dataset', related_name='%(app_label)s_%(class)s_second')

    last_updated = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        abstract = True
        unique_together = (
            ('dataset_1', 'dataset_2',),
        )


class DatasetDataDistance(DatasetDistance):
    '''
    Distance between datasets considering data values.
    '''
    distance = models.FloatField()


class DatasetMetadataDistance(DatasetDistance):
    '''
    Distance between datasets considering metadata values.
    '''
    distance = models.FloatField()


class ExperimentDistance(models.Model):
    '''
    Base model for experiment distances.
    '''
    experiment_1 = models.ForeignKey(
        'Experiment', related_name='%(app_label)s_%(class)s_first')
    experiment_2 = models.ForeignKey(
        'Experiment', related_name='%(app_label)s_%(class)s_second')

    last_updated = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        abstract = True
        unique_together = (
            ('experiment_1', 'experiment_2',),
        )


class ExperimentDataDistance(ExperimentDistance):
    '''
    Distance between experiments considering data values.
    '''
    distance = models.FloatField()


class ExperimentMetadataDistance(ExperimentDistance):
    '''
    Distance between experiments considering metadata values.
    '''
    distance = models.FloatField()


class MetaPlot(models.Model):
    locus_group = models.ForeignKey('LocusGroup')
    dataset = models.ForeignKey('Dataset')

    metaplot = JSONField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (
            ('locus_group', 'dataset',),
        )


class Ontology(models.Model):
    name = models.CharField(unique=True, max_length=128)
    ontology_type = models.CharField(choices=ONTOLOGY_TYPES, max_length=64)
    obo_file = models.FilePathField(path=settings.DATA_PATH)
    ac_file = models.FilePathField(path=settings.DATA_PATH)

    last_updated = models.DateTimeField(auto_now=True, null=True)

    def get_ontology_object(self):
        return OntologyObject(self.obo_file, self.ac_file,
                              ontology_type=self.ontology_type)


class UserToExperimentSimilarity(models.Model):
    user = models.ForeignKey('MyUser')
    experiment = models.ForeignKey('Experiment')

    score = models.FloatField()

    last_updated = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        unique_together = (
            ('user', 'experiment',),
        )
