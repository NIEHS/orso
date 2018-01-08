import operator

from django.contrib import admin
from django.core.urlresolvers import reverse

from network import models


def obj_link(model, description=None, reverse_url=None, pk_attr=None,
             name_attr=None):

    if not description:
        description = model.title()
    if not reverse_url:
        reverse_url = 'admin:network_{}_change'.format(model)
    if not pk_attr:
        pk_attr = '{}.pk'.format(model)
    if not name_attr:
        name_attr = '{}.name'.format(model)

    def f(self, obj):
        try:
            pk = operator.attrgetter(pk_attr)(obj)
            url = reverse(reverse_url, args=(pk, ))
        except AttributeError:
            return '-'

        try:
            name = operator.attrgetter(name_attr)(obj)
        except AttributeError:
            return '<a href="{}">{}</a>'.format(url, pk)
        else:
            return '{} (ID: <a href="{}">{}</a>)'.format(name, url, pk)

    f.allow_tags = True
    f.short_description = description
    return f


def obj_count(model, obj_lookup, description=None):
    if not description:
        description = model._meta.verbose_name_plural.title()

    def f(self, obj):
        return model.objects.filter(**{obj_lookup: obj}).distinct().count()
    f.short_description = description
    return f


# TabularInline objects
class AnnotationInline(admin.TabularInline):
    model = models.Annotation
    show_change_link = True
    extra = 0

    fields = [
        'id', 'name', '_genes', '_transcripts', '_enhancers', 'last_updated',
    ]
    readonly_fields = fields

    _enhancers = obj_count(models.Enhancer, 'annotation')
    _genes = obj_count(models.Gene, 'annotation')
    _transcripts = obj_count(models.Transcript, 'gene__annotation')


class DataInline(admin.TabularInline):
    extra = 0
    show_change_link = True

    fields = [
        'id', '_assembly', '_locus_group', '_locus_group_type', 'last_updated',
    ]
    readonly_fields = fields

    _assembly = obj_link(
        'assembly',
        pk_attr='locus_group.assembly.pk',
        name_attr='locus_group.assembly.name'
    )
    _locus_group = obj_link(
        'locus_group', reverse_url='admin:network_locusgroup_change')

    def _locus_group_type(self, obj):
        return obj.locus_group.group_type


class IntersectionInline(DataInline):
    model = models.DatasetIntersectionJson
    exclude = ['intersection_values']


class MetaPlotInline(DataInline):
    model = models.MetaPlot
    exclude = ['metaplot']


class DatasetInline(admin.TabularInline):
    model = models.Dataset
    show_change_link = True
    extra = 0

    fields = ['id', 'name', '_assembly', 'processed']
    readonly_fields = fields

    _assembly = obj_link('assembly')


class LocusInline(admin.TabularInline):
    model = models.Locus
    show_change_link = True
    extra = 0

    fields = [
        'id', '_locus_group', '_assembly', '_type', '_strand', 'chromosome',
        'regions',
    ]
    readonly_fields = fields

    _assembly = obj_link(
        'assembly',
        pk_attr='group.assembly.pk',
        name_attr='group.assembly.name',
    )
    _locus_group = obj_link(
        'locus_group',
        reverse_url='admin:network_locusgroup_change',
        pk_attr='group.pk',
        name_attr='group.name'
    )

    def _strand(self, obj):
        if obj.group.group_type == 'enhancer':
            return 'N/A'
        else:
            return obj.strand

    def _type(self, obj):
        return obj.group.group_type


class TranscriptInline(admin.TabularInline):
    model = models.Transcript
    extra = 0
    show_change_link = True

    readonly_fields = ['name', 'chromosome', 'strand', 'start', 'end', 'exons']
    fields = readonly_fields


class TransformedValuesInline(admin.TabularInline):
    model = models.PCATransformedValues
    extra = 0
    show_change_link = True

    readonly_fields = ['id', 'dataset', 'last_updated']
    fields = readonly_fields


# ModelAdmin objects
@admin.register(models.Annotation)
class AnnotationAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'name', '_assembly', 'last_updated', '_genes', '_transcripts',
        '_enhancers',
    ]
    fields = [
        'id', 'name', '_assembly', 'last_updated', '_genes', '_transcripts',
        '_enhancers',
    ]
    readonly_fields = [
        'id', '_assembly', 'last_updated', '_genes', '_transcripts',
        '_enhancers',
    ]

    _assembly = obj_link('assembly')
    _enhancers = obj_count(models.Enhancer, 'annotation')
    _genes = obj_count(models.Gene, 'annotation')
    _transcripts = obj_count(models.Transcript, 'gene__annotation')


@admin.register(models.Assembly)
class AssemblyAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'name', '_annotations', '_datasets', '_experiments',
        'last_updated',
    ]
    fields = [
        'id', 'name', 'chromosome_sizes',
    ]
    readonly_fields = [
        'id',
    ]
    inlines = [AnnotationInline]

    _annotations = obj_count(models.Annotation, 'assembly')
    _datasets = obj_count(models.Dataset, 'assembly')
    _experiments = obj_count(models.Experiment, 'dataset__assembly')


@admin.register(models.Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'name', '_owners', '_experiment', '_assembly', 'created',
        'last_updated', 'processed',
    ]
    fields = [
        'id', 'name', 'slug', 'consortial_id', 'created', 'last_updated',
        'processed', '_assembly', '_experiment', 'description',
        'ambiguous_url', 'plus_url', 'minus_url',
    ]
    readonly_fields = [
        'id', '_assembly', '_experiment', 'created', 'last_updated',
    ]
    inlines = [IntersectionInline, MetaPlotInline]

    _experiment = obj_link('experiment')
    _assembly = obj_link('assembly')

    def _owners(self, obj):
        owners = obj.experiment.owners.all()
        if owners:
            owner_links = []
            for owner in owners:
                url = reverse('admin:network_myuser_change',
                              args=(owner.pk, ))
                owner_links.append(
                    '<a href="{}">{}</a>'.format(url, owner.user.username))
            return ', '.join(owner_links)
        else:
            return 'None'
    _owners.allow_tags = True


@admin.register(models.DatasetDataDistance,
                models.DatasetMetadataDistance)
class DatasetDistanceAdmin(admin.ModelAdmin):
    list_display = [
        'id', '_dataset_1', '_dataset_2', '_assembly', '_experiment_type',
    ]
    fields = [
        'id', '_dataset_1', '_dataset_2', '_assembly', '_experiment_type',
        'distance',
    ]
    readonly_fields = [
        'id', '_dataset_1', '_dataset_2', '_assembly', '_experiment_type'
    ]

    _assembly = obj_link(
        'assembly',
        pk_attr='dataset_1.assembly.pk',
        name_attr='dataset_1.assembly.name',
    )
    _dataset_1 = obj_link(
        'dataset_1', reverse_url='admin:network_dataset_change')
    _dataset_2 = obj_link(
        'dataset_2', reverse_url='admin:network_dataset_change')
    _experiment_type = obj_link(
        'experiment_type',
        description='Experiment Type',
        reverse_url='admin:network_experimenttype_change',
        pk_attr='dataset_1.experiment.experiment_type.pk',
        name_attr='dataset_1.experiment.experiment_type.name',
    )


@admin.register(models.Enhancer)
class EnhancerAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'name', '_assembly', '_annotation', 'chromosome', '_strand',
        'start', 'end',
    ]
    fields = [
        'id', 'name', '_assembly', '_annotation', 'chromosome', '_strand',
        'start', 'end',
    ]
    readonly_fields = fields
    inlines = [LocusInline]

    _assembly = obj_link(
        'assembly',
        pk_attr='annotation.assembly.pk',
        name_attr='annotation.assembly.name',
    )
    _annotation = obj_link('annotation')

    def _strand(self, obj):
        return 'N/A'


@admin.register(models.Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'name', '_owners', '_project', '_datasets', 'created',
        'last_updated',
    ]
    fields = [
        'id', 'name', 'slug', 'consortial_id', '_project', '_owners',
        'cell_type', 'target', 'processed', 'description',
    ]
    readonly_fields = [
        'id', '_project', '_owners',
    ]
    inlines = [DatasetInline]

    _datasets = obj_count(models.Dataset, 'experiment')
    _project = obj_link('project')

    def _owners(self, obj):
        owners = obj.owners.all()
        if owners:
            owner_links = []
            for owner in owners:
                url = reverse('admin:network_myuser_change',
                              args=(owner.pk, ))
                owner_links.append(
                    '<a href="{}">{}</a>'.format(url, owner.user.username))
            return ', '.join(owner_links)
        else:
            return 'None'
    _owners.allow_tags = True


@admin.register(models.ExperimentDataDistance,
                models.ExperimentMetadataDistance)
class ExperimentDistanceAdmin(admin.ModelAdmin):
    list_display = [
        'id', '_experiment_1', '_experiment_2', '_assemblies',
        '_experiment_type',
    ]
    fields = [
        'id', '_experiment_1', '_experiment_2', '_assemblies',
        '_experiment_type', 'distance',
    ]
    readonly_fields = fields

    _experiment_1 = obj_link(
        'experiment_1', reverse_url='admin:network_experiment_change')
    _experiment_2 = obj_link(
        'experiment_2', reverse_url='admin:network_experiment_change')
    _experiment_type = obj_link(
        'experiment_type',
        description='Experiment Type',
        reverse_url='admin:network_experimenttype_change',
        pk_attr='experiment_1.experiment_type.pk',
        name_attr='experiment_1.experiment_type.name',
    )

    def _assemblies(self, obj):
        assemblies_1 = models.Assembly.objects.filter(
            dataset__experiment=obj.experiment_1).distinct()
        assemblies_2 = models.Assembly.objects.filter(
            dataset__experiment=obj.experiment_2).distinct()
        shared_assemblies = assemblies_1 & assemblies_2

        assembly_links = []
        for assembly in shared_assemblies:
            url = reverse('admin:network_assembly_change',
                          args=(assembly.pk, ))
            assembly_links.append(
                '<a href="{}">{}</a>'.format(url, assembly.name))

        return ', '.join(assembly_links)
    _assemblies.allow_tags = True


@admin.register(models.ExperimentType)
class ExperimentTypeAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'name', 'short_name',
    ]


@admin.register(models.Gene)
class GeneAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'name', '_transcripts', '_assembly', '_annotation',
    ]
    fields = [
        'id', 'name', '_assembly', '_annotation', '_selected_transcript',
    ]
    readonly_fields = fields
    inlines = [TranscriptInline]

    _annotation = obj_link('annotation')
    _assembly = obj_link(
        'assembly',
        pk_attr='annotation.assembly.pk',
        name_attr='annotation.assembly.name',
    )
    _selected_transcript = obj_link(
        'transcript',
        description='Selected Transcript',
        pk_attr='selected_transcript.pk',
        name_attr='selected_transcript.name',
    )
    _transcripts = obj_count(models.Transcript, 'gene')


@admin.register(models.DatasetIntersectionJson)
class IntersectionAdmin(admin.ModelAdmin):
    list_display = ['id', '_dataset', '_locus_group', '_assembly',
                    'locus_group_type', 'last_updated']
    readonly_fields = ['id', '_dataset', '_locus_group', '_assembly',
                       'locus_group_type', 'last_updated']
    fields = readonly_fields + ['intersection_values']

    _assembly = obj_link(
        'assembly',
        pk_attr='locus_group.assembly.pk',
        name_attr='locus_group.assembly.name',
    )
    _dataset = obj_link('dataset')
    _locus_group = obj_link(
        'locus_group', reverse_url='admin:network_locusgroup_change')

    def locus_group_type(self, obj):
        return obj.locus_group.group_type


@admin.register(models.Locus)
class LocusAdmin(admin.ModelAdmin):
    list_display = [
        'id', '_transcript', '_enhancer', '_locus_group', '_assembly', '_type',
        'chromosome', '_strand', 'regions'
    ]
    fields = [
        'id', '_transcript', '_enhancer', '_locus_group', '_assembly', '_type',
        'chromosome', '_strand', 'regions'
    ]
    readonly_fields = fields

    _assembly = obj_link(
        'assembly',
        pk_attr='group.assembly.pk',
        name_attr='group.assembly.name'
    )
    _enhancer = obj_link('enhancer')
    _locus_group = obj_link(
        'group',
        description='Locus Group',
        reverse_url='admin:network_locusgroup_change',
    )
    _transcript = obj_link('transcript')

    def _strand(self, obj):
        if obj.group.group_type == 'enhancer':
            return 'N/A'
        else:
            return obj.strand

    def _type(self, obj):
        return obj.group.group_type


@admin.register(models.LocusGroup)
class LocusGroupAdmin(admin.ModelAdmin):
    list_display = [
        'id', '_assembly', 'group_type', '_loci',
    ]
    fields = [
        'id', '_assembly', 'group_type', '_loci',
    ]
    readonly_fields = fields

    _assembly = obj_link('assembly')
    _loci = obj_count(models.Locus, 'group')


@admin.register(models.MetaPlot)
class MetaPlotAdmin(admin.ModelAdmin):
    list_display = [
        'id', '_dataset', '_assembly', '_locus_group', '_locus_group_type',
        'last_updated',
    ]
    fields = [
        'id', '_dataset', '_assembly', '_locus_group', '_locus_group_type',
        'last_updated',
    ]
    readonly_fields = fields

    _assembly = obj_link(
        'assembly',
        pk_attr='locus_group.assembly.pk',
        name_attr='locus_group.assembly.name',
    )
    _dataset = obj_link('dataset')
    _locus_group = obj_link(
        'locus_group', reverse_url='admin:network_locusgroup_change')

    def _locus_group_type(self, obj):
        return obj.locus_group.group_type


@admin.register(models.MyUser)
class MyUserAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'user', '_datasets', '_date_joined', '_last_login',
    ]
    fields = [
        'id', 'user', '_datasets', '_date_joined', '_last_login',
    ]
    readonly_fields = fields

    def _datasets(self, obj):
        return models.Dataset.objects.filter(
            experiment__owners__in=[obj]).count()

    def _date_joined(self, obj):
        return obj.user.date_joined

    def _last_login(self, obj):
        return obj.user.last_login


@admin.register(models.Ontology)
class OntologyAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'name', 'ontology_type', 'last_updated',
    ]
    fields = [
        'id', 'name', 'ontology_type', 'last_updated', 'obo_file', 'ac_file',
    ]
    readonly_fields = fields


@admin.register(models.PCA)
class PCAAdmin(admin.ModelAdmin):
    list_display = [
        'id', '_assembly', '_locus_group', '_locus_group_type',
        '_experiment_type', '_transformed_datasets', 'last_updated',
    ]
    fields = [
        'id', '_assembly', '_locus_group', '_locus_group_type',
        '_experiment_type', '_transformed_datasets', 'last_updated',
    ]
    readonly_fields = fields
    exclude = ['plot']
    inlines = [TransformedValuesInline]

    _assembly = obj_link(
        'assembly',
        pk_attr='locus_group.assembly.pk',
        name_attr='locus_group.assembly.name',
    )
    _experiment_type = obj_link(
        'experiment_type',
        description='Experiment Type',
        reverse_url='admin:network_experimenttype_change'
    )
    _locus_group = obj_link(
        'locus_group', reverse_url='admin:network_locusgroup_change')
    _transformed_datasets = obj_count(
        models.PCATransformedValues, 'pca', description='Transformed datasets')

    def _locus_group_type(self, obj):
        return obj.locus_group.group_type


@admin.register(models.PCATransformedValues)
class PCATransformedValuesAdmin(admin.ModelAdmin):
    list_display = ['id', '_pca', '_dataset', 'last_updated']
    readonly_fields = ['id', '_pca', '_dataset', 'transformed_values']
    fields = readonly_fields

    _pca = obj_link('pca')
    _dataset = obj_link('dataset')


@admin.register(models.Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', '_owners', '_experiments', '_datasets',
                    'created', 'last_updated']

    def _owners(self, obj):
        owners = obj.owners.all()
        if owners:
            return ', '.join([owner.user.username for owner in owners])
        else:
            return 'None'

    _experiments = obj_count(models.Experiment, 'project')
    _datasets = obj_count(models.Dataset, 'experiment__project')


@admin.register(models.Transcript)
class TranscriptAdmin(admin.ModelAdmin):
    list_display = [
        'name', '_gene', '_assembly', '_annotation', 'chromosome', 'strand',
        'start', 'end',
    ]
    fields = [
        '_assembly', '_annotation', '_gene', 'chromosome', 'strand', 'start',
        'end', 'exons', 'is_selected',
    ]
    readonly_fields = fields
    inlines = [LocusInline]

    _annotation = obj_link(
        'annotation',
        pk_attr='gene.annotation.pk',
        name_attr='gene.annotation.name',
    )
    _assembly = obj_link(
        'assembly',
        pk_attr='gene.annotation.assembly.pk',
        name_attr='gene.annotation.assembly.name',
    )
    _gene = obj_link('gene')

    def is_selected(self, obj):
        return models.Gene.objects.filter(selected_transcript=obj).exists()
