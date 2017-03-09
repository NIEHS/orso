from django import forms
from django.db.models import Q
from selectable.forms import AutoCompleteWidget

from . import lookups
# from selectable import forms as selectable

# from . import models
# from analysis import metaplot


class ExperimentFilterForm(forms.Form):

    order_choices = [
        ('correlation_rank', 'correlation'),
        ('metadata_rank', 'metadata'),
        ('collaborative_rank', 'collaboration'),
    ]

    order = forms.MultipleChoiceField(
        label='Recommendation criteria',
        choices=order_choices,
        widget=forms.CheckboxSelectMultiple(),
        initial=[c[0] for c in order_choices],
        required=False)

    name = forms.CharField(
        label='Name',
        widget=AutoCompleteWidget(lookups.NameLookup),
        required=False)

    description = forms.CharField(
        label='Description',
        widget=AutoCompleteWidget(lookups.DescriptionLookup),
        required=False)

    data_type = forms.CharField(
        label='Data type',
        help_text="ex: ChIP-seq",
        widget=AutoCompleteWidget(lookups.DataTypeLookup),
        required=False)

    cell_type = forms.CharField(
        label='Cell type',
        help_text='ex: K562',
        widget=AutoCompleteWidget(lookups.CellTypeLookup),
        required=False)

    assembly = forms.CharField(
        label='Assembly',
        help_text='ex: hg19',
        widget=AutoCompleteWidget(lookups.AssemblyLookup),
        required=False)

    target = forms.CharField(
        label='Target',
        help_text='Target antibody',
        widget=AutoCompleteWidget(lookups.TargetLookup),
        required=False)

    paginate_by = forms.IntegerField(
        label='Items per page',
        min_value=1,
        initial=10,
        max_value=10000,
        required=False)

    def get_query(self):

        name = self.cleaned_data.get('name')
        description = self.cleaned_data.get('description')
        data_type = self.cleaned_data.get('data_type')
        cell_type = self.cleaned_data.get('cell_type')
        assembly = self.cleaned_data.get('assembly')

        query = Q()
        if name:
            query &= Q(name__icontains=name)
        if description:
            query &= Q(description__icontains=name)
        if data_type:
            query &= Q(data_type__icontains=data_type)
        if cell_type:
            query &= Q(cell_type__icontains=cell_type)
        if assembly:
            query &= Q(dataset__assembly__name__icontains=assembly)

        return query

    def get_order(self):
        return self.cleaned_data.get('order')


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
