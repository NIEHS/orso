from selectable.base import ModelLookup
from selectable.registry import registry

from . import models


class DistinctStringLookup(ModelLookup):
    """
    Return distinct strings for a single CharField in a model
    """
    distinct_field = None

    def get_query(self, request, term):
        return self.get_queryset()\
            .filter(**{self.distinct_field + "__icontains": term})\
            .order_by(self.distinct_field)\
            .distinct(self.distinct_field)

    def get_item_value(self, item):
        return getattr(item, self.distinct_field)

    def get_item_label(self, item):
        return self.get_item_value(item)


class DataTypeLookup(DistinctStringLookup):
    model = models.Experiment
    distinct_field = 'data_type'

registry.register(DataTypeLookup)
