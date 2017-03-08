from django import forms
from django.db.models import Q
from selectable.forms import AutoCompleteWidget

from . import lookups
# from selectable import forms as selectable

# from . import models
# from analysis import metaplot


class ExperimentFilterForm(forms.Form):

    data_type = forms.CharField(
        label='Data type',
        help_text="ex: ChIP-seq",
        widget=AutoCompleteWidget(lookups.DataTypeLookup),
        required=False)

    paginate_by = forms.IntegerField(
        label='Items per page',
        min_value=1,
        initial=10,
        max_value=10000,
        required=False)

    def get_query(self):

        data_type = self.cleaned_data.get('data_type')

        query = Q()
        if data_type:
            query &= Q(data_type=data_type)

        return query


class ExperimentForm(forms.ModelForm):
    pass
    # URL_HELP = 'URL for downloading user-dataset, must be publicly available without authentication.'  # noqa
    #
    # ambiguous_url = forms.URLField(
    #     required=False,
    #     label='URL (strands unspecified)',
    #     help_text=URL_HELP)
    # plus_url = forms.URLField(
    #     required=False,
    #     label='URL (plus-strand)',
    #     help_text=URL_HELP)
    # minus_url = forms.URLField(
    #     required=False,
    #     label='URL (minus-strand)',
    #     help_text=URL_HELP)
    #
    # class Meta:
    #     model = models.Dataset
    #     fields = (
    #         'data_type', 'cell_type', 'target', 'description',
    #         'ambiguous_url', 'plus_url', 'minus_url', 'name', 'assembly'
    #     )
