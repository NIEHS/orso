import random
import requests
import numpy
import json
import string
import os
from collections import defaultdict

import networkx as nx
from django.conf import settings
from django.contrib.postgres.fields import JSONField, ArrayField
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q
from django.urls import reverse
from fa2 import ForceAtlas2
from nltk.corpus import stopwords
from picklefield.fields import PickledObjectField
from scipy.stats import variation as coeff_variance

from network.tasks.analysis.ontology import Ontology as OntologyObject
from network.tasks.analysis.metaplot import generate_metaplot_bed
from network.tasks.analysis.coverage import generate_locusgroup_bed
from network.management.commands.update_dendrogram import \
    call_update_dendrogram
from network.tasks.analysis.network import update_organism_network
from network.tasks.utils import get_exp_tag

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

    def save(self, **kwargs):
        self.clean()
        return super().save(**kwargs)

    def clean(self):
        super().clean()
        if any([
            self.dataset_1 is not None and self.dataset_2 is None,
            self.dataset_2 is not None and self.dataset_1 is None,
        ]):
            raise ValidationError('Similarity requires paired datasets.')


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
    organism = models.ForeignKey('Organism', models.PROTECT)

    cell_type = models.CharField(max_length=128)
    target = models.CharField(max_length=128, blank=True)
    name = models.CharField(max_length=128)
    slug = models.CharField(max_length=128)
    description = models.TextField(blank=True)
    created = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True, null=True)

    consortial_id = models.CharField(
        verbose_name='Repository ID', max_length=128, null=True, blank=True)
    consortial_url = models.URLField(
        verbose_name='Repository URL', null=True, blank=True)

    use_default_color = models.BooleanField(default=True)
    color = models.CharField(max_length=7, blank=True, null=True, default=None)

    public = models.BooleanField(default=True)

    processed = models.BooleanField(default=False)
    revoked = models.BooleanField(default=False)

    def __str__(self):
        return self.name

    class Meta:
        get_latest_by = 'created'
        unique_together = ('project', 'consortial_id', 'experiment_type',
                           'cell_type', 'target')

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

        exp_tag = get_exp_tag(self)

        in_plot = False
        for node in plot['nodes']:
            if node['exp_tag'] == exp_tag:
                in_plot = True
                exp_node = node['id']

        if not in_plot:
            return None

        connected_pks = set([exp_node])
        for edge in plot['edges']:
            if edge['source'] == exp_node:
                connected_pks.add(edge['target'])
            elif edge['target'] == exp_node:
                connected_pks.add(edge['source'])

        g = nx.Graph()

        for pk in connected_pks:
            g.add_node(pk)
        for edge in plot['edges']:
            if all([
                edge['source'] in connected_pks,
                edge['target'] in connected_pks,
            ]):
                g.add_edge(edge['source'], edge['target'])

        fa2 = ForceAtlas2()
        try:
            positions = fa2.forceatlas2_networkx_layout(
                g, pos=None, iterations=50)
        except ZeroDivisionError:
            positions = dict()
            for pk in connected_pks:
                positions[pk] = (random.random(), random.random())

        nodes = []
        edges = []

        for node in plot['nodes']:
            if node['id'] in connected_pks:
                node['x'] = positions[node['id']][0]
                node['y'] = positions[node['id']][1]
                if node['id'] == exp_node:
                    node['selected'] = 'True'
                nodes.append(node)

        for edge in plot['edges']:
            if all([
                edge['source'] in connected_pks,
                edge['target'] in connected_pks,
            ]):
                edges.append(edge)

        return {
            'nodes': nodes,
            'edges': edges,
        }


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

    predicted_cell_type = \
        models.CharField(max_length=128, null=True, blank=True)
    predicted_target = \
        models.CharField(max_length=128, null=True, blank=True)

    predicted_cell_type_json = JSONField(blank=True, null=True)
    predicted_target_json = JSONField(blank=True, null=True)

    class Meta:
        get_latest_by = 'created'
        unique_together = ('experiment', 'consortial_id', 'ambiguous_url',
                           'plus_url', 'minus_url')

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
                settings.BIGWIG_TEMP_DIR, '{}.plus.bigWig'.format(str(self.pk)))  # noqa
            minus_local_path = os.path.join(
                settings.BIGWIG_TEMP_DIR, '{}.minus.bigWig'.format(str(self.pk)))  # noqa
            return {
                'ambiguous': None,
                'plus': plus_local_path,
                'minus': minus_local_path,
            }
        else:
            ambiguous_local_path = os.path.join(
                settings.BIGWIG_TEMP_DIR, '{}.ambig.bigWig'.format(str(self.pk)))  # noqa
            return {
                'ambiguous': ambiguous_local_path,
                'plus': None,
                'minus': None,
            }

    def get_predicted_classes(self, predicted_field, threshold=0.5):
        class_set = set()

        if predicted_field == 'cell_type':
            prediction_json = self.predicted_cell_type_json
        elif predicted_field == 'target':
            prediction_json = self.predicted_target_json
        else:
            raise ValueError('Improper predicted_field value.')

        try:
            predictions = json.loads(prediction_json)
            for predicted_class, prediction_value in predictions:
                if prediction_value > threshold:
                    class_set.add(predicted_class)
        except TypeError:
            pass

        return class_set

    def get_predicted_cell_types(self, **kwargs):
        return self.get_predicted_classes('cell_type', **kwargs)

    def get_predicted_targets(self, **kwargs):
        return self.get_predicted_classes('target', **kwargs)

    def get_filtered_intersection(self):

        relevant_regions = self.experiment.experiment_type.relevant_regions
        dij = DatasetIntersectionJson.objects.get(
            dataset=self,
            locus_group__group_type=relevant_regions,
        )
        try:
            pca = PCA.objects.get(
                locus_group__group_type=relevant_regions,
                locus_group__assembly=self.assembly,
                experiment_type=self.experiment.experiment_type,
            )
        except PCA.DoesNotExist:
            return None
        else:

            order = PCALocusOrder.objects.filter(pca=pca).order_by('order')
            loci = [x.locus for x in order]

            intersection_values = json.loads(dij.intersection_values)

            locus_values = dict()
            for val, pk in zip(
                intersection_values['normalized_values'],
                intersection_values['locus_pks']
            ):
                locus_values[pk] = val

            normalized_values = []
            for locus in loci:
                try:
                    normalized_values.append(locus_values[locus.pk])
                except IndexError:
                    normalized_values.append(0)

            return normalized_values

    def set_filtered_intersection(self):

        intersection = self.get_filtered_intersection()
        if intersection:
            self.filtered_intersection_values = intersection
            self.save()

    # def get_network(self):
    #     network = DatasetNetwork.objects.get(dataset=self)
    #     return json.loads(network.network_plot)


class Organism(models.Model):
    name = models.CharField(unique=True, max_length=32)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


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


class NeuralNetwork(models.Model):
    NN_MODEL_DIR = 'neural_networks/'

    locus_group = models.ForeignKey('LocusGroup')
    experiment_type = models.ForeignKey('ExperimentType')
    metadata_field = models.CharField(
        max_length=24,
        choices=(('cell_type', 'cell_type'), ('target', 'target'),),
    )

    neural_network_file = models.FileField(
        null=True, blank=True, max_length=256, upload_to=NN_MODEL_DIR)
    neural_network_scaler = PickledObjectField(null=True, blank=True)
    neural_network_label_binarizer = PickledObjectField(null=True, blank=True)

    loss = models.FloatField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    training_history = PickledObjectField(null=True, blank=True)

    last_updated = models.DateTimeField(auto_now=True, null=True)

    def _get_path(self, file_name):
        return os.path.join(settings.MEDIA_ROOT, self.NN_MODEL_DIR,
                            file_name.format(str(self.pk)))

    def get_nn_model_path(self):
        return self._get_path('{}.hd5')

    def get_accuracy_plot_path(self):
        return self._get_path('{}_training_accuracy.png')

    def get_loss_plot_path(self):
        return self._get_path('{}_training_loss.png')


class PCA(models.Model):
    '''
    scikit-learn PCA object. Used to transform locus intersections and
    find pairwise distances between datasets.
    '''
    NN_MODEL_DIR = 'nn_models/'

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
    neural_network_file = models.FileField(
        null=True, blank=True, max_length=256, upload_to=NN_MODEL_DIR)
    neural_network_scaler = PickledObjectField(null=True, blank=True)

    last_updated = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return str(self.pk)

    class Meta:
        verbose_name = 'PCA'
        verbose_name_plural = 'PCAs'

    def get_nn_model_path(self):
        return os.path.join(settings.MEDIA_ROOT, self.NN_MODEL_DIR,
                            '{}.hd5'.format(str(self.pk)))


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

    def get_norm_vector(self):
        intersection_dict = json.loads(self.intersection_values)

        locus_values = dict()
        for val, locus_pk in zip(
            intersection_dict['normalized_values'],
            intersection_dict['locus_pks'],
        ):
            locus_values[locus_pk] = val

        vector = []
        for locus in Locus.objects.filter(
                group=self.locus_group).order_by('pk'):
            try:
                vector.append(locus_values[locus.pk])
            except KeyError:
                vector.append(0.0)

        return vector

    def get_filtered_vector(self):

        try:
            pca = PCA.objects.get(
                locus_group=self.locus_group,
                experiment_type=self.dataset.experiment.experiment_type,
            )
        except PCA.DoesNotExist:
            return None
        else:

            order = PCALocusOrder.objects.filter(pca=pca).order_by('order')
            loci = [x.locus for x in order]

            intersection_values = json.loads(self.intersection_values)

            locus_values = dict()
            for val, pk in zip(
                intersection_values['normalized_values'],
                intersection_values['locus_pks']
            ):
                locus_values[pk] = val

            filtered_values = []
            for locus in loci:
                try:
                    filtered_values.append(locus_values[locus.pk])
                except IndexError:
                    filtered_values.append(0)

            return filtered_values


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
