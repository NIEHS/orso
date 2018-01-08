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


class AllExpLookup(DistinctStringLookup):
    def get_query(self, request, term):
        return self.get_queryset() \
            .filter(**{self.distinct_field + "__icontains": term}) \
            .order_by(self.distinct_field) \
            .distinct(self.distinct_field)


class AllExpNameLookup(AllExpLookup):
    model = models.Experiment
    distinct_field = 'name'


class AllExpDescriptionLookup(AllExpLookup):
    model = models.Experiment
    distinct_field = 'description'


class AllExpCellTypeLookup(AllExpLookup):
    model = models.Experiment
    distinct_field = 'cell_type'


class AllExpTargetLookup(AllExpLookup):
    model = models.Experiment
    distinct_field = 'target'


class RecExpLookup(DistinctStringLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        user_experiments = models.Experiment.objects.filter(
            owners=my_user)
        query = Q(network_experimentdatadistance_first__experiment_2__in=user_experiments)  # noqa
        query &= ~Q(network_experimentdatadistance_second__experiment_1__in=user_experiments)  # noqa
        return self.get_queryset() \
            .filter(**{self.distinct_field + "__icontains": term}) \
            .filter(query) \
            .order_by(self.distinct_field) \
            .distinct(self.distinct_field)


class RecExpNameLookup(RecExpLookup):
    model = models.Experiment
    distinct_field = 'name'


class RecExpDescriptionLookup(RecExpLookup):
    model = models.Experiment
    distinct_field = 'description'


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
            models.Assembly.objects.filter(dataset__experiment=exp)
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


class ExperimentTypeLookup(ModelLookup):
    model = models.Experiment

    def get_item_value(self, item):
        value = item.experiment_type.name
        return value

    def get_item_label(self, item):
        return self.get_item_value(item)


class AllExpTypeLookup(ExperimentTypeLookup):
    def get_query(self, request, term):
        return self.get_queryset() \
            .order_by('experiment_type__name') \
            .distinct('experiment_type__name')


class RecExpTypeLookup(ExperimentTypeLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        user_experiments = models.Experiment.objects.filter(
            owners=my_user)
        query = Q(network_experimentdatadistance_first__experiment_2__in=user_experiments)  # noqa
        query &= ~Q(network_experimentdatadistance_second__experiment_1__in=user_experiments)  # noqa
        query &= Q(experiment_type__name__icontains=term)
        return self.get_queryset() \
            .filter(query) \
            .order_by('experiment_type__name') \
            .distinct('experiment_type__name')


class PerExpTypeLookup(ExperimentTypeLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        return self.get_queryset()\
            .filter(**{'experiment_type__name__icontains': term,
                    'owners__in': [my_user]})\
            .order_by('experiment_type__name')\
            .distinct('experiment_type__name')


class FavExpTypeLookup(ExperimentTypeLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        return self.get_queryset()\
            .filter(**{'experiment_type__name__icontains': term,
                    'experimentfavorite__owner': my_user})\
            .order_by('experiment_type__name')\
            .distinct('experiment_type__name')


class AssemblyLookup(ModelLookup):
    model = models.Dataset

    def get_item_value(self, item):
        value = item.assembly.name
        return value

    def get_item_label(self, item):
        return self.get_item_value(item)


class AllExpAssemblyLookup(AssemblyLookup):
    def get_query(self, request, term):
        return self.get_queryset() \
            .order_by('assembly__name') \
            .distinct('assembly__name')


class RecExpAssemblyLookup(AssemblyLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        user_experiments = models.Experiment.objects.filter(
            owners=my_user)
        query = Q(experiment__network_experimentdatadistance_first__experiment_2__in=user_experiments)  # noqa
        query &= ~Q(experiment__network_experimentdatadistance_second__experiment_1__in=user_experiments)  # noqa
        query &= Q(assembly__name__icontains=term)
        return self.get_queryset() \
            .filter(query) \
            .order_by('assembly__name') \
            .distinct('assembly__name')


class PerExpAssemblyLookup(AssemblyLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        return self.get_queryset()\
            .filter(**{'assembly__name__icontains': term,
                    'experiment__owners__in': [my_user]})\
            .order_by('assembly__name')\
            .distinct('assembly__name')


class FavExpAssemblyLookup(AssemblyLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        return self.get_queryset()\
            .filter(**{'assembly__name__icontains': term,
                    'experiment__experimentfavorite__owner': my_user})\
            .order_by('assembly__name')\
            .distinct('assembly__name')


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
        query |= Q(experiment_type__name__icontains=term)
        query |= Q(cell_type__icontains=term)
        query |= Q(description__icontains=term)
        query |= Q(target__icontains=term)
        query |= Q(dataset__assembly__name__icontains=term)

        return query

    def get_item_value(self, item):
        values = []
        spacer = ' | '

        assemblies = set()
        for a in models.Assembly.objects.filter(dataset__experiment=item):  # noqa
            assemblies.add(a.name)

        if item.name:
            values.append(item.name)
        if item.experiment_type:
            values.append(item.experiment_type.name)
        if item.cell_type:
            values.append(item.cell_type)
        if item.target:
            values.append(item.target)
        if assemblies:
            values.append(spacer.join(list(assemblies)))
        # if item.description:
        #     values.append(item.description)

        return spacer.join(values)

    def get_item_label(self, item):
        return self.get_item_value(item)


class AllExpSearchLookup(ExperimentSearchLookup):
    def get_query(self, request, term):
        return self.get_queryset()\
            .order_by('name')\
            .distinct()


class RecExpSearchLookup(ExperimentSearchLookup):
    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        user_experiments = models.Experiment.objects.filter(
            owners=my_user)
        query = self.get_base_query(request, term)
        query &= Q(network_experimentdatadistance_first__experiment_2__in=user_experiments)  # noqa
        query &= ~Q(network_experimentdatadistance_second__experiment_1__in=user_experiments)  # noqa
        return self.get_queryset()\
            .filter(query)\
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
            models.Assembly.objects.filter(dataset__experiment=exp)
        query = Q()
        for a in assemblies:
            query |= Q(dataset__assembly=a)
        base_query = self.get_base_query(request, term)
        return self.get_queryset()\
            .filter(query & base_query)\
            .exclude(pk=exp.pk)\
            .order_by('name')\
            .distinct()

registry.register(AllExpNameLookup)
registry.register(AllExpDescriptionLookup)
registry.register(AllExpTypeLookup)
registry.register(AllExpCellTypeLookup)
registry.register(AllExpTargetLookup)
registry.register(AllExpAssemblyLookup)
registry.register(AllExpSearchLookup)

registry.register(RecExpNameLookup)
registry.register(RecExpDescriptionLookup)
registry.register(RecExpTypeLookup)
registry.register(RecExpCellTypeLookup)
registry.register(RecExpTargetLookup)
registry.register(RecExpAssemblyLookup)
registry.register(RecExpSearchLookup)

registry.register(PerExpNameLookup)
registry.register(PerExpDescriptionLookup)
registry.register(PerExpTypeLookup)
registry.register(PerExpCellTypeLookup)
registry.register(PerExpTargetLookup)
registry.register(PerExpAssemblyLookup)
registry.register(PerExpSearchLookup)

registry.register(FavExpNameLookup)
registry.register(FavExpDescriptionLookup)
registry.register(FavExpTypeLookup)
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
