from django import forms
from django.db.models import Q
from selectable.forms import AutoCompleteWidget

from . import lookups
# from selectable import forms as selectable

from . import models
# from django.forms.models import inlineformset_factory
# from analysis import metaplot


class ExperimentFilterForm(forms.Form):

    search = forms.CharField(
        label='Search',
        widget=AutoCompleteWidget(lookups.RecommendationSearchLookup),
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

        search = self.cleaned_data.get('search')

        search_query = Q()
        if search:
            search_query |= Q(name__icontains=search)
            search_query |= Q(description__icontains=search)
            search_query |= Q(data_type__icontains=search)
            search_query |= Q(cell_type__icontains=search)
            search_query |= Q(dataset__assembly__name__icontains=search)

        name = self.cleaned_data.get('name')
        description = self.cleaned_data.get('description')
        data_type = self.cleaned_data.get('data_type')
        cell_type = self.cleaned_data.get('cell_type')
        assembly = self.cleaned_data.get('assembly')

        filter_query = Q()
        if name:
            filter_query &= Q(name__icontains=name)
        if description:
            filter_query &= Q(description__icontains=description)
        if data_type:
            filter_query &= Q(data_type__icontains=data_type)
        if cell_type:
            filter_query &= Q(cell_type__icontains=cell_type)
        if assembly:
            filter_query &= Q(dataset__assembly__name__icontains=assembly)

        return Q(search_query & filter_query)


class ExperimentRecFilterForm(ExperimentFilterForm):
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

    field_order = [
        'order'
    ]

    def get_order(self):
        return self.cleaned_data.get('order')


class ExperimentForm(forms.ModelForm):

    class Meta:
        model = models.Experiment
        fields = (
            'data_type', 'cell_type', 'target', 'description', 'name'
        )


class DatasetForm(forms.ModelForm):

    class Meta:
        model = models.Dataset
        fields = (
            'name', 'assembly', 'description', 'ambiguous_url', 'plus_url',
            'minus_url'
        )

    def clean(self):
        cleaned_data = super().clean()

        ambiguous_url = cleaned_data.get('ambiguous_url')
        plus_url = cleaned_data.get('plus_url')
        minus_url = cleaned_data.get('minus_url')

        error_text = 'An ambiguous URL may not be included with stranded URLs.'

        if ambiguous_url:
            if plus_url or minus_url:
                raise forms.ValidationError(error_text)
        elif plus_url and minus_url:
            if ambiguous_url:
                raise forms.ValidationError(error_text)
        else:
            raise forms.ValidationError(
                'Either ambiguous or stranded URLs required.')
