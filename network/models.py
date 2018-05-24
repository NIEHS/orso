import requests
import pyBigWig
import numpy
import math
import json
import string
import os
from collections import defaultdict

from django.conf import settings
from django.contrib.postgres.fields import JSONField, ArrayField
from django.db import models
from django.db.models import Q
from django.urls import reverse
from nltk.corpus import stopwords
from picklefield.fields import PickledObjectField
from scipy.stats import variation as coeff_variance

from analysis.ontology import Ontology as OntologyObject
from analysis.metaplot import generate_metaplot_bed
from analysis.transcript_coverage import generate_locusgroup_bed
from network.management.commands.update_dendrogram import \
    call_update_dendrogram
from network.tasks.network import update_organism_network
from network.tasks.utils import rgba_to_string, string_to_rgba

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
    followed_users = models.ManyToManyField(
        'MyUser', symmetrical=False, blank=True, through='Follow')

    favorite_experiments = models.ManyToManyField(
        'Experiment', blank=True, related_name='favorited',
        through='Favorite')
    primary_data_recommedations = models.ManyToManyField(
        'Experiment', blank=True, related_name='primary_rec',
        through='PrimaryDataRec', through_fields=('user', 'experiment'))
    metadata_recommendations = models.ManyToManyField(
        'Experiment', blank=True, related_name='metadata_rec',
        through='MetadataRec', through_fields=('user', 'experiment'))

    public = models.BooleanField(default=True)

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL)
    slug = models.CharField(
        max_length=128)

    def __str__(self):
        return self.user.username

    class Meta:
        verbose_name = 'MyUser'
        verbose_name_plural = 'MyUsers'

    def is_public(self):
        return self.public

    def is_owned(self, user):
        return self.user == user

    def get_user_favorites_count(self):
        pass

    def get_data_favorites_count(self):
        pass

    def get_user_details(self, my_user=None):
        detail = dict()

        detail['username'] = self.user.username
        detail['pk'] = self.pk

        detail['experiment_number'] = \
            Experiment.objects.filter(owners=self).distinct().count()
        detail['data_favorited_by_number'] = \
            Favorite.objects.filter(experiment__owners=self).distinct().count()
        detail['data_favorite_number'] = \
            Favorite.objects.filter(user=self).count()

        detail['user_followed_by_number'] = \
            Follow.objects.filter(followed=self).count()
        detail['user_following_number'] = \
            Follow.objects.filter(following=self).count()

        if my_user:
            detail['is_followed'] = self.is_followed(my_user)

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

    # def get_experiment_counts(self):
    #     favorite_experiment_counts = \
    #         len(ExperimentFavorite.objects.filter(owner=self))
    #     personal_experiment_counts = \
    #         len(Experiment.objects.filter(owners__in=[self]))
    #
    #     return {
    #         'favorite_experiment_counts': favorite_experiment_counts,
    #         'personal_experiment_counts': personal_experiment_counts,
    #     }

    def get_urls(self):
        add_favorite = reverse('api:user-follow', kwargs={'pk': self.pk})
        remove_favorite = \
            reverse('api:user-stop-following', kwargs={'pk': self.pk})
        hide_recommendation = \
            reverse('api:user-hide-recommendation', kwargs={'pk': self.pk})
        detail = reverse('user', kwargs={'pk': self.user.pk})

        return {
            'add_favorite': add_favorite,
            'remove_favorite': remove_favorite,
            'hide_recommendation': hide_recommendation,
            'detail': detail,
        }

    def is_followed(self, my_user):
        if Follow.objects.filter(following=my_user, followed=self).exists():
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


class Follow(models.Model):
    following = models.ForeignKey('MyUser', related_name='following')
    followed = models.ForeignKey('MyUser', related_name='followed')

    created = models.DateTimeField(auto_now_add=True, null=True)


class Favorite(models.Model):
    user = models.ForeignKey('MyUser')
    experiment = models.ForeignKey('Experiment')

    created = models.DateTimeField(auto_now_add=True, null=True)


class Access(models.Model):
    user = models.ForeignKey('MyUser')
    access_count = models.PositiveIntegerField(default=0)

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        abstract = True


class ExperimentAccess(Access):
    experiment = models.ForeignKey('Experiment')

    class Meta:
        unique_together = ('user', 'experiment')


class DatasetAccess(Access):
    dataset = models.ForeignKey('Dataset')

    class Meta:
        unique_together = ('user', 'dataset')


class PrimaryDataRec(models.Model):
    user = models.ForeignKey('MyUser')
    experiment = models.ForeignKey(
        'Experiment', related_name='%(class)s_recommended')
    dataset = models.ForeignKey(
        'Dataset', related_name='%(class)s_recommended')

    personal_experiment = models.ForeignKey(
        'Experiment', related_name='%(class)s_owned')
    personal_dataset = models.ForeignKey(
        'Dataset', related_name='%(class)s_owned')

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)


class MetadataRec(models.Model):
    user = models.ForeignKey('MyUser')
    experiment = models.ForeignKey(
        'Experiment', related_name='%(class)s_recommended')

    personal_experiment = models.ForeignKey(
        'Experiment', related_name='%(class)s_owned')

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)


class Recommendation(models.Model):
    user = models.ForeignKey('MyUser')

    recommended_experiment = models.ForeignKey(
        'Experiment', related_name='recommended_experiment',
    )
    recommended_dataset = models.ForeignKey(
        'Dataset', related_name='recommended_dataset',
        null=True, blank=True,
    )

    referring_experiment = models.ForeignKey(
        'Experiment', related_name='referring_experiment',
        null=True, blank=True,
    )
    referring_dataset = models.ForeignKey(
        'Dataset', related_name='referring_dataset',
        null=True, blank=True,
    )
    referring_user = models.ForeignKey(
        'MyUser', related_name='referring_user',
        null=True, blank=True,
    )

    choices = [
        ('primary', 'primary'),
        ('metadata', 'metadata'),
        ('user', 'user'),
    ]
    rec_type = models.CharField(choices=choices, max_length=32)

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)

    def get_recommendation_tag(self):

        if self.rec_type == 'primary':
            exp_url = reverse(
                'experiment', args=[self.referring_experiment.pk])
            return {
                'tag': 'Similar data values to ',
                'target_name': self.referring_experiment.name,
                'target_url': exp_url,
            }

        elif self.rec_type == 'metadata':
            exp_url = reverse(
                'experiment', args=[self.referring_experiment.pk])
            return {
                'tag': 'Similar metadata to ',
                'target_name': self.referring_experiment.name,
                'target_url': exp_url,
            }

        elif self.rec_type == 'user':
            user_url = reverse(
                'user', args=[self.referring_user.pk])
            return {
                'tag': 'Interactions by user ',
                'target_name': self.referring_user.user.username,
                'target_url': user_url,
            }


class Similarity(models.Model):
    experiment_1 = models.ForeignKey(
        'Experiment', related_name='sim_experiment_1')
    experiment_2 = models.ForeignKey(
        'Experiment', related_name='sim_experiment_2')

    dataset_1 = models.ForeignKey(
        'Dataset', related_name='dataset_1',
        null=True, blank=True,
    )
    dataset_2 = models.ForeignKey(
        'Dataset', related_name='dataset_2',
        null=True, blank=True,
    )

    choices = [
        ('primary', 'primary'),
        ('metadata', 'metadata'),
    ]
    sim_type = models.CharField(choices=choices, max_length=32)

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        unique_together = (
            'sim_type', 'experiment_1', 'experiment_2', 'dataset_1',
            'dataset_2')


class Project(models.Model):
    owners = models.ManyToManyField('MyUser', blank=True)

    name = models.CharField(max_length=128)
    description = models.TextField(blank=True)

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return self.name


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

    public = models.BooleanField(default=True)

    processed = models.BooleanField(default=False)
    revoked = models.BooleanField(default=False)

    def __str__(self):
        return self.name

    class Meta:
        get_latest_by = 'created'
        unique_together = ('project', 'consortial_id')

    def delete(self):
        organism = Organism.objects.get(assembly__dataset__experiment=self)
        exp_type = self.experiment_type
        my_users = list(self.owners.all())

        super().delete()

        # Update user networks and dendrograms after delete
        for my_user in my_users:
            update_organism_network.si(
                organism.pk, exp_type.pk, my_user_pk=my_user.pk).delay()
            call_update_dendrogram.si(
                organism.pk, exp_type.pk, my_user_pk=my_user.pk).delay()

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

    def is_public(self):
        return self.public

    def is_owned(self, user):
        return self.owners.filter(pk=user.pk).exists()

    def is_favorite(self, my_user):
        # if ExperimentFavorite.objects.filter(owner=my_user, favorite=self).exists():  # noqa
        #     return 'true'
        # else:
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

    def get_recommendation_tags(self, my_user):
        query = Q(recommended_experiment=self)
        query &= Q(user=my_user)
        query &= ~Q(recommended_experiment__owners=my_user)
        recs = Recommendation.objects.filter(query)

        tags = [rec.get_recommendation_tag() for rec in recs]
        tags = set([(tag['tag'], tag['target_name'], tag['target_url'])
                   for tag in tags])
        tags = sorted(list(tags), key=lambda x: (x[0], x[1], x[2]))
        tags = [{'tag': tag[0], 'target_name': tag[1], 'target_url': tag[2]}
                for tag in tags]

        return tags

    def get_network(self, my_user=None):
        '''
        Grabs the organism-level network plot. Adds a hidden 'center' node
        used to move the camera to capture all nodes connected to the
        experiment. Also sets nodes that are not connected to be transparent.
        '''

        organism = Organism.objects.filter(
            assembly__dataset__experiment=self)[0]
        try:
            network = OrganismNetwork.objects.get(
                organism=organism, experiment_type=self.experiment_type,
                my_user=my_user)
        except OrganismNetwork.DoesNotExist:
            network = OrganismNetwork.objects.get(
                organism=organism, experiment_type=self.experiment_type)

        plot = json.loads(network.network_plot)

        in_plot = False
        for node in plot['nodes']:
            if node['id'] == self.pk:
                in_plot = True

        if not in_plot:
            return None

        connection_pks = Experiment.objects.filter(
            sim_experiment_1__experiment_2=self
        ).distinct().values_list('pk', flat=True)
        pk_set = set([self.pk] + list(connection_pks))

        max_total_x = float('-inf')
        min_total_x = float('inf')
        max_total_y = float('-inf')
        min_total_y = float('inf')

        max_field_x = float('-inf')
        min_field_x = float('inf')
        max_field_y = float('-inf')
        min_field_y = float('inf')

        for node in plot['nodes']:

            max_total_x = max(max_total_x, node['x'])
            min_total_x = min(min_total_x, node['x'])
            max_total_y = max(max_total_y, node['y'])
            min_total_y = min(min_total_y, node['y'])

            if node['id'] in pk_set:
                max_field_x = max(max_field_x, node['x'])
                min_field_x = min(min_field_x, node['x'])
                max_field_y = max(max_field_y, node['y'])
                min_field_y = min(min_field_y, node['y'])
            else:
                rgba = string_to_rgba(node['color'])
                rgba[3] = 0.2
                node['color'] = rgba_to_string(rgba)

        for edge in plot['edges']:
            if all([
                edge['source'] != self.pk,
                edge['target'] != self.pk,
            ]):
                rgba = string_to_rgba(edge['color'])
                rgba[3] = 0.2
                edge['color'] = rgba_to_string(rgba)

        x_position = numpy.mean([max_field_x, min_field_x])
        y_position = numpy.mean([max_field_y, min_field_y])
        zoom_ratio = max(
            (max_field_x - min_field_x) / (max_total_x - min_total_x),
            (max_field_y - min_field_y) / (max_total_y - min_total_y),
        )
        if zoom_ratio == 0:
            zoom_ratio = 1

        center_added = False
        for node in plot['nodes']:
            if node['id'] == 'center':
                node['x'] = x_position
                node['y'] = y_position
                node['color'] = rgba_to_string((0, 0, 0, 0))
                center_added = True

        if not center_added:
            plot['nodes'].append({
                'x': x_position,
                'y': y_position,
                'id': 'center',
                'color': rgba_to_string((0, 0, 0, 0)),
                'size': 0,
            })

        plot['camera'] = {
            'zoom_ratio': zoom_ratio,
        }

        return plot


class ExperimentNetwork(models.Model):
    experiment = models.ForeignKey('Experiment')

    network_plot = JSONField()
    last_updated = models.DateTimeField(auto_now=True)


class Dataset(models.Model):
    assembly = models.ForeignKey('Assembly')
    experiment = models.ForeignKey('Experiment', blank=True, null=True)

    name = models.CharField(max_length=128)
    slug = models.CharField(max_length=128)
    created = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)
    description = models.TextField(blank=True)
    consortial_id = models.CharField(max_length=128, null=True, default=None)

    ambiguous_url = models.URLField(null=True, blank=True)
    plus_url = models.URLField(null=True, blank=True)
    minus_url = models.URLField(null=True, blank=True)

    processed = models.BooleanField(default=False)
    revoked = models.BooleanField(default=False)

    class Meta:
        get_latest_by = 'created'
        unique_together = ('experiment', 'consortial_id')

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse('dataset', kwargs={'pk': self.pk})

    def get_favorites_count(self):
        pass

    def is_public(self):
        return self.experiment.public

    def is_owned(self, user):
        return self.experiment.owners.filter(pk=user.pk).exists()

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

    def generate_local_bigwig_paths(self):
        if self.is_stranded():
            plus_local_path = os.path.join(
                settings.BIGWIG_TEMP_DIR, os.path.basename(self.plus_url))
            minus_local_path = os.path.join(
                settings.BIGWIG_TEMP_DIR, os.path.basename(self.minus_url))
            return {
                'ambiguous': None,
                'plus': plus_local_path,
                'minus': minus_local_path,
            }
        else:
            ambiguous_local_path = os.path.join(
                settings.BIGWIG_TEMP_DIR, os.path.basename(self.ambiguous_url))
            return {
                'ambiguous': ambiguous_local_path,
                'plus': None,
                'minus': None,
            }

    # def get_network(self):
    #     network = DatasetNetwork.objects.get(dataset=self)
    #     return json.loads(network.network_plot)


class DatasetNetwork(models.Model):
    dataset = models.ForeignKey('Dataset')

    network_plot = JSONField()
    last_updated = models.DateTimeField(auto_now=True)


class Organism(models.Model):
    name = models.CharField(unique=True, max_length=32)
    last_updated = models.DateTimeField(auto_now=True)


class OrganismNetwork(models.Model):
    organism = models.ForeignKey('Organism')
    experiment_type = models.ForeignKey('ExperimentType')
    my_user = models.ForeignKey('MyUser', blank=True, null=True)

    network_plot = JSONField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('organism', 'experiment_type', 'my_user')


class Dendrogram(models.Model):
    organism = models.ForeignKey('Organism')
    experiment_type = models.ForeignKey('ExperimentType')
    my_user = models.ForeignKey('MyUser', blank=True, null=True)

    dendrogram_plot = JSONField()
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('organism', 'experiment_type', 'my_user')


class Assembly(models.Model):
    organism = models.ForeignKey('Organism')

    name = models.CharField(unique=True, max_length=32)
    chromosome_sizes = JSONField(blank=True, null=True)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name_plural = 'Assemblies'

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

    def __str__(self):
        return self.name


class PCA(models.Model):
    '''
    scikit-learn PCA object. Used to transform locus intersections and
    find pairwise distances between datasets.
    '''
    locus_group = models.ForeignKey('LocusGroup')
    experiment_type = models.ForeignKey('ExperimentType')

    selected_loci = models.ManyToManyField(
        'Locus', through='PCALocusOrder')
    transformed_datasets = models.ManyToManyField(
        'Dataset', through='PCATransformedValues')

    plot = JSONField(null=True, blank=True)
    pca = PickledObjectField(null=True, blank=True)
    covariation_matrix = PickledObjectField(null=True, blank=True)
    inverse_covariation_matrix = PickledObjectField(null=True, blank=True)

    neural_network = PickledObjectField(null=True, blank=True)
    neural_network_scaler = PickledObjectField(null=True, blank=True)

    last_updated = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return str(self.pk)

    class Meta:
        verbose_name = 'PCA'
        verbose_name_plural = 'PCAs'


class PCATransformedValues(models.Model):
    '''
    Used to store PCA-transformed dataset values.
    '''
    pca = models.ForeignKey('PCA')
    dataset = models.ForeignKey('Dataset')

    transformed_values = ArrayField(models.FloatField(), size=3)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str(self.pk)

    class Meta:
        verbose_name = 'PCA Transformed Values'
        verbose_name_plural = verbose_name


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

    def __str__(self):
        return str(self.pk)

    class Meta:
        verbose_name = 'Locus'
        verbose_name_plural = 'Loci'

    def get_name(self):
        if self.transcript:
            return self.transcript.gene.name
        elif self.enhancer:
            return self.enhancer.name
        else:
            return None


class FeatureAttributes(models.Model):
    locus_group = models.ForeignKey('LocusGroup')
    experiment_type = models.ForeignKey('ExperimentType')

    feature_attributes = JSONField()

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)


class LocusGroup(models.Model):
    assembly = models.ForeignKey('Assembly')
    group_type = models.CharField(choices=LOCUS_GROUP_TYPES, max_length=32)

    intersection_bed_path = models.FilePathField(blank=True, null=True)
    metaplot_bed_path = models.FilePathField(blank=True, null=True)

    def __str__(self):
        return str(self.pk)

    def create_and_set_intersection_bed(self):
        # if not os.path.exists(settings.INTERSECTION_BED_DIR):
        os.makedirs(settings.INTERSECTION_BED_DIR, exist_ok=True)
        path = os.path.join(
            settings.INTERSECTION_BED_DIR,
            '{}-{}-{}-intersection.bed'.format(
                self.pk, self.assembly.name, self.group_type)
        )
        with open(path, 'w') as BED:
            generate_locusgroup_bed(self, BED)
        self.intersection_bed_path = path
        self.save()

    def create_and_set_metaplot_bed(self):
        # if not os.path.exists(settings.METAPLOT_BED_DIR):
        os.makedirs(settings.METAPLOT_BED_DIR, exist_ok=True)
        path = os.path.join(
            settings.METAPLOT_BED_DIR,
            '{}-{}-{}-metaplot.bed'.format(
                self.pk, self.assembly.name, self.group_type)
        )
        with open(path, 'w') as BED:
            generate_metaplot_bed(self, BED)
        self.metaplot_bed_path = path
        self.save()


class Gene(models.Model):
    annotation = models.ForeignKey('Annotation')
    selected_transcript = models.ForeignKey(
        'Transcript', related_name='selecting', blank=True, null=True)

    name = models.CharField(
        max_length=32)

    def __str__(self):
        return self.name

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

    def __str__(self):
        return self.name

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

    def __str__(self):
        return self.name


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

    def __str__(self):
        return str(self.pk)

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

    def __str__(self):
        return str(self.pk)

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
    recommended = models.BooleanField()


class DatasetMetadataDistance(DatasetDistance):
    '''
    Distance between datasets considering metadata values.
    '''
    distance = models.FloatField()
    recommended = models.BooleanField()


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

    def __str__(self):
        return str(self.pk)

    class Meta:
        unique_together = (
            ('locus_group', 'dataset',),
        )
        verbose_name = 'Metaplot'
        verbose_name_plural = 'Metaplots'


class FeatureValues(models.Model):
    locus_group = models.ForeignKey('LocusGroup')
    dataset = models.ForeignKey('Dataset')

    feature_values = JSONField()

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return str(self.pk)

    class Meta:
        unique_together = (
            ('locus_group', 'dataset',),
        )
        verbose_name = 'FeatureValues'
        verbose_name_plural = 'FeatureValues'


class Ontology(models.Model):
    name = models.CharField(unique=True, max_length=128)
    ontology_type = models.CharField(choices=ONTOLOGY_TYPES, max_length=64)
    obo_file = models.FilePathField(path=settings.DATA_PATH)
    ac_file = models.FilePathField(path=settings.DATA_PATH)

    last_updated = models.DateTimeField(auto_now=True, null=True)

    class Meta:
        verbose_name = 'Ontology'
        verbose_name_plural = 'Ontologies'

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


class AdminNotification(models.Model):
    created = models.DateTimeField(auto_now_add=True, null=True)
    message = models.TextField()


class Activity(models.Model):
    user = models.ForeignKey('MyUser', blank=True, null=True)

    created_experiment = models.ForeignKey(
        'Experiment', related_name='created_experiment',
        blank=True, null=True)
    favorited_experiment = models.ForeignKey(
        'Experiment', related_name='favorited_experiment',
        blank=True, null=True)
    followed_user = models.ForeignKey(
        'MyUser', related_name='followed_user',
        blank=True, null=True)

    administrative_action = models.BooleanField(default=False)

    created = models.DateTimeField(auto_now_add=True, null=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)
