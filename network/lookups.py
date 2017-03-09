from selectable.base import ModelLookup
from selectable.registry import registry

from . import models


class DistinctStringLookup(ModelLookup):
    """
    Return distinct strings for a single CharField in a model
    """
    distinct_field = None

    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        return self.get_queryset()\
            .filter(**{self.distinct_field + "__icontains": term,
                    'experimentrecommendation__owner': my_user})\
            .order_by(self.distinct_field)\
            .distinct(self.distinct_field)

    def get_item_value(self, item):
        return getattr(item, self.distinct_field)

    def get_item_label(self, item):
        return self.get_item_value(item)


class DataTypeLookup(DistinctStringLookup):
    model = models.Experiment
    distinct_field = 'data_type'


class AssemblyLookup(ModelLookup):
    #  TODO: double check when Experiments exist with multiple assemblies
    model = models.Experiment
    displayed = set()

    def get_query(self, request, term):
        my_user = models.MyUser.objects.get(user=request.user)
        return self.get_queryset()\
            .filter(**{'dataset__assembly__name__icontains': term,
                    'experimentrecommendation__owner': my_user})\
            .order_by('dataset__assembly__name')\
            .distinct('dataset__assembly__name')

    def get_item_value(self, item):
        value = models.Dataset.objects.get(experiment=item).assembly.name
        return value

    def get_item_label(self, item):
        return self.get_item_value(item)

registry.register(DataTypeLookup)
registry.register(AssemblyLookup)
