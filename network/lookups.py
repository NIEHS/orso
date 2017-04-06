from selectable.base import ModelLookup
from selectable.registry import registry
from django.db.models import Q

from . import models


class DistinctStringLookup(ModelLookup):
    """
    Return distinct strings for a single CharField in a model
    """
    distinct_field = None

    def get_item_value(self, item):
        return getattr(item, self.distinct_field)

    def get_item_label(self, item):
        return self.get_item_value(item)


class RecExpLookup(DistinctStringLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        return self.get_queryset()\
            .filter(**{self.distinct_field + "__icontains": term,
                    'experimentrecommendation__owner': my_user})\
            .order_by(self.distinct_field)\
            .distinct(self.distinct_field)


class RecExpNameLookup(RecExpLookup):
    model = models.Experiment
    distinct_field = 'name'


class RecExpDescriptionLookup(RecExpLookup):
    model = models.Experiment
    distinct_field = 'description'


class RecExpDataTypeLookup(RecExpLookup):
    model = models.Experiment
    distinct_field = 'data_type'


class RecExpCellTypeLookup(RecExpLookup):
    model = models.Experiment
    distinct_field = 'cell_type'


class RecExpTargetLookup(RecExpLookup):
    model = models.Experiment
    distinct_field = 'target'


class PerExpLookup(DistinctStringLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        return self.get_queryset()\
            .filter(**{self.distinct_field + "__icontains": term,
                    'owners__in': [my_user]})\
            .order_by(self.distinct_field)\
            .distinct(self.distinct_field)


class PerExpNameLookup(PerExpLookup):
    model = models.Experiment
    distinct_field = 'name'


class PerExpDescriptionLookup(PerExpLookup):
    model = models.Experiment
    distinct_field = 'description'


class PerExpDataTypeLookup(PerExpLookup):
    model = models.Experiment
    distinct_field = 'data_type'


class PerExpCellTypeLookup(PerExpLookup):
    model = models.Experiment
    distinct_field = 'cell_type'


class PerExpTargetLookup(PerExpLookup):
    model = models.Experiment
    distinct_field = 'target'


class FavExpLookup(DistinctStringLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        return self.get_queryset()\
            .filter(**{self.distinct_field + "__icontains": term,
                    'experimentfavorite__owner': my_user})\
            .order_by(self.distinct_field)\
            .distinct(self.distinct_field)


class FavExpNameLookup(FavExpLookup):
    model = models.Experiment
    distinct_field = 'name'


class FavExpDescriptionLookup(FavExpLookup):
    model = models.Experiment
    distinct_field = 'description'


class FavExpDataTypeLookup(FavExpLookup):
    model = models.Experiment
    distinct_field = 'data_type'


class FavExpCellTypeLookup(FavExpLookup):
    model = models.Experiment
    distinct_field = 'cell_type'


class FavExpTargetLookup(FavExpLookup):
    model = models.Experiment
    distinct_field = 'target'


class SimExpLookup(DistinctStringLookup):
    def get_query(self, request, term):
        exp = models.Experiment.objects.get(pk=request.GET['pk'])
        assemblies = \
            models.GenomeAssembly.objects.filter(dataset__experiment=exp)
        query = Q()
        for a in assemblies:
            query |= Q(dataset__assembly=a)
        return self.get_queryset()\
            .filter(**{self.distinct_field + "__icontains": term})\
            .filter(query)\
            .exclude(pk=exp.pk)\
            .order_by(self.distinct_field)\
            .distinct(self.distinct_field)


class SimExpNameLookup(SimExpLookup):
    model = models.Experiment
    distinct_field = 'name'


class SimExpDescriptionLookup(SimExpLookup):
    model = models.Experiment
    distinct_field = 'description'


class SimExpDataTypeLookup(SimExpLookup):
    model = models.Experiment
    distinct_field = 'data_type'


class SimExpCellTypeLookup(SimExpLookup):
    model = models.Experiment
    distinct_field = 'cell_type'


class SimExpTargetLookup(SimExpLookup):
    model = models.Experiment
    distinct_field = 'target'


class AssemblyLookup(ModelLookup):
    #  TODO: double check when Experiments exist with multiple assemblies
    model = models.Experiment

    def get_item_value(self, item):
        value = models.Dataset.objects.get(experiment=item).assembly.name
        return value

    def get_item_label(self, item):
        return self.get_item_value(item)


class RecExpAssemblyLookup(AssemblyLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        return self.get_queryset()\
            .filter(**{'dataset__assembly__name__icontains': term,
                    'experimentrecommendation__owner': my_user})\
            .order_by('dataset__assembly__name')\
            .distinct('dataset__assembly__name')


class PerExpAssemblyLookup(AssemblyLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        return self.get_queryset()\
            .filter(**{'dataset__assembly__name__icontains': term,
                    'owners__in': [my_user]})\
            .order_by('dataset__assembly__name')\
            .distinct('dataset__assembly__name')


class FavExpAssemblyLookup(AssemblyLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        return self.get_queryset()\
            .filter(**{'dataset__assembly__name__icontains': term,
                    'experimentfavorite__owner': my_user})\
            .order_by('dataset__assembly__name')\
            .distinct('dataset__assembly__name')


class SimExpAssemblyLookup(AssemblyLookup):
    def get_query(self, request, term):
        pk = request.GET['pk']
        return self.get_queryset()\
            .filter(**{'dataset__assembly__name__icontains': term})\
            .exclude(pk=pk)\
            .order_by('dataset__assembly__name')\
            .distinct('dataset__assembly__name')


class ExperimentSearchLookup(ModelLookup):
    model = models.Experiment

    def get_base_query(self, request, term):
        query = Q()
        query |= Q(name__icontains=term)
        query |= Q(data_type__icontains=term)
        query |= Q(cell_type__icontains=term)
        query |= Q(description__icontains=term)
        query |= Q(target__icontains=term)
        query |= Q(dataset__assembly__name__icontains=term)

        return query

    def get_item_value(self, item):
        values = []
        spacer = ' | '

        assemblies = set()
        for a in models.GenomeAssembly.objects.filter(dataset__experiment=item):  # noqa
            assemblies.add(a.name)

        if item.name:
            values.append(item.name)
        if item.data_type:
            values.append(item.data_type)
        if item.cell_type:
            values.append(item.cell_type)
        if item.target:
            values.append(item.target)
        if assemblies:
            values.append(spacer.join(list(assemblies)))
        if item.description:
            values.append(item.description)

        return spacer.join(values)

    def get_item_label(self, item):
        return self.get_item_value(item)


class RecExpSearchLookup(ExperimentSearchLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        query = self.get_base_query(request, term)
        return self.get_queryset()\
            .filter(Q(experimentrecommendation__owner=my_user) & query)\
            .order_by('name')\
            .distinct()


class PerExpSearchLookup(ExperimentSearchLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        query = self.get_base_query(request, term)
        return self.get_queryset()\
            .filter(Q(owners__in=[my_user]) & query)\
            .order_by('name')\
            .distinct()


class FavExpSearchLookup(ExperimentSearchLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        query = self.get_base_query(request, term)
        return self.get_queryset()\
            .filter(Q(experimentfavorite__owner=my_user) & query)\
            .order_by('name')\
            .distinct()


class SimExpSearchLookup(ExperimentSearchLookup):
    def get_query(self, request, term):
        exp = models.Experiment.objects.get(pk=request.GET['pk'])
        assemblies = \
            models.GenomeAssembly.objects.filter(dataset__experiment=exp)
        query = Q()
        for a in assemblies:
            query |= Q(dataset__assembly=a)
        base_query = self.get_base_query(request, term)
        return self.get_queryset()\
            .filter(query & base_query)\
            .exclude(pk=exp.pk)\
            .order_by('name')\
            .distinct()

registry.register(RecExpNameLookup)
registry.register(RecExpDescriptionLookup)
registry.register(RecExpDataTypeLookup)
registry.register(RecExpCellTypeLookup)
registry.register(RecExpTargetLookup)
registry.register(RecExpAssemblyLookup)
registry.register(RecExpSearchLookup)

registry.register(PerExpNameLookup)
registry.register(PerExpDescriptionLookup)
registry.register(PerExpDataTypeLookup)
registry.register(PerExpCellTypeLookup)
registry.register(PerExpTargetLookup)
registry.register(PerExpAssemblyLookup)
registry.register(PerExpSearchLookup)

registry.register(FavExpNameLookup)
registry.register(FavExpDescriptionLookup)
registry.register(FavExpDataTypeLookup)
registry.register(FavExpCellTypeLookup)
registry.register(FavExpTargetLookup)
registry.register(FavExpAssemblyLookup)
registry.register(FavExpSearchLookup)

registry.register(SimExpNameLookup)
registry.register(SimExpDescriptionLookup)
registry.register(SimExpDataTypeLookup)
registry.register(SimExpCellTypeLookup)
registry.register(SimExpTargetLookup)
registry.register(SimExpAssemblyLookup)
registry.register(SimExpSearchLookup)
