import requests
from django import forms
from django.contrib.auth.models import User
from django.db.models import Q
from django.forms.widgets import TextInput
from selectable.forms import AutoCompleteWidget

from . import lookups

from . import models


class UserFilterForm(forms.Form):

    search = forms.CharField(
        label='Search',
        required=False)

    paginate_by = forms.IntegerField(
        label='Items per page',
        min_value=1,
        initial=10,
        max_value=10000,
        required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['search'].widget = \
            AutoCompleteWidget(lookups.UserNameLookup)

    def get_query(self):

        search = self.cleaned_data.get('search')
        search_query = Q(user__username__contains=search)

        return search_query


class ExperimentFilterForm(forms.Form):

    search = forms.CharField(
        label='Search',
        required=False)

    name = forms.CharField(
        label='Name',
        required=False)

    description = forms.CharField(
        label='Description',
        required=False)

    data_type = forms.CharField(
        label='Data type',
        help_text="ex: ChIP-seq",
        required=False)

    cell_type = forms.CharField(
        label='Cell/tissue type',
        help_text='ex: K562',
        required=False)

    assembly = forms.CharField(
        label='Assembly',
        help_text='ex: hg19',
        required=False)

    target = forms.CharField(
        label='Target',
        help_text='Target antibody',
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
            search_query |= Q(experiment_type__name__icontains=search)
            search_query |= Q(cell_type__icontains=search)
            search_query |= Q(dataset__assembly__name__icontains=search)
            search_query |= Q(target__icontains=search)

        name = self.cleaned_data.get('name')
        description = self.cleaned_data.get('description')
        data_type = self.cleaned_data.get('data_type')
        cell_type = self.cleaned_data.get('cell_type')
        assembly = self.cleaned_data.get('assembly')
        target = self.cleaned_data.get('target')

        filter_query = Q()
        if name:
            filter_query &= Q(name__icontains=name)
        if description:
            filter_query &= Q(description__icontains=description)
        if data_type:
            filter_query &= Q(experiment_type__name__icontains=data_type)
        if cell_type:
            filter_query &= Q(cell_type__icontains=cell_type)
        if assembly:
            filter_query &= Q(dataset__assembly__name__icontains=assembly)
        if target:
            filter_query &= Q(target__icontains=target)

        return Q(search_query & filter_query)


class AllExperimentFilterForm(ExperimentFilterForm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.fields['search'].widget = \
        #     AutoCompleteWidget(lookups.AllExpSearchLookup)
        self.fields['name'].widget = \
            AutoCompleteWidget(lookups.AllExpNameLookup)
        # self.fields['description'].widget = \
        #     AutoCompleteWidget(lookups.AllExpDescriptionLookup)
        self.fields['data_type'].widget = \
            AutoCompleteWidget(lookups.AllExpTypeLookup)
        self.fields['cell_type'].widget = \
            AutoCompleteWidget(lookups.AllExpCellTypeLookup)
        self.fields['assembly'].widget = \
            AutoCompleteWidget(lookups.AllExpAssemblyLookup)
        self.fields['target'].widget = \
            AutoCompleteWidget(lookups.AllExpTargetLookup)


class RecommendedExperimentFilterForm(ExperimentFilterForm):
    rec_type_choices = [
        ('primary', 'Primary Data'),
        ('metadata', 'Metadata'),
        ('user', 'User Interactions'),
    ]

    rec_type = forms.MultipleChoiceField(
        label='Recommendation criteria',
        choices=rec_type_choices,
        initial=[c[0] for c in rec_type_choices],
        required=False,
        widget=forms.CheckboxSelectMultiple(),
    )

    field_order = [
        'rec_type'
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.fields['search'].widget = \
        #     AutoCompleteWidget(lookups.RecExpSearchLookup)
        self.fields['name'].widget = \
            AutoCompleteWidget(lookups.RecExpNameLookup)
        # self.fields['description'].widget = \
        #     AutoCompleteWidget(lookups.RecExpDescriptionLookup)
        self.fields['data_type'].widget = \
            AutoCompleteWidget(lookups.RecExpTypeLookup)
        self.fields['cell_type'].widget = \
            AutoCompleteWidget(lookups.RecExpCellTypeLookup)
        self.fields['assembly'].widget = \
            AutoCompleteWidget(lookups.RecExpAssemblyLookup)
        self.fields['target'].widget = \
            AutoCompleteWidget(lookups.RecExpTargetLookup)

    def get_rec_type(self):
        return self.cleaned_data.get('rec_type')


class PersonalExperimentFilterForm(ExperimentFilterForm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.fields['search'].widget = \
        #     AutoCompleteWidget(lookups.PerExpSearchLookup)
        self.fields['name'].widget = \
            AutoCompleteWidget(lookups.PerExpNameLookup)
        # self.fields['description'].widget = \
        #     AutoCompleteWidget(lookups.PerExpDescriptionLookup)
        self.fields['data_type'].widget = \
            AutoCompleteWidget(lookups.PerExpTypeLookup)
        self.fields['cell_type'].widget = \
            AutoCompleteWidget(lookups.PerExpCellTypeLookup)
        self.fields['assembly'].widget = \
            AutoCompleteWidget(lookups.PerExpAssemblyLookup)
        self.fields['target'].widget = \
            AutoCompleteWidget(lookups.PerExpTargetLookup)


class FavoriteExperimentFilterForm(ExperimentFilterForm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.fields['search'].widget = \
        #     AutoCompleteWidget(lookups.FavExpSearchLookup)
        self.fields['name'].widget = \
            AutoCompleteWidget(lookups.FavExpNameLookup)
        # self.fields['description'].widget = \
        #     AutoCompleteWidget(lookups.FavExpDescriptionLookup)
        self.fields['data_type'].widget = \
            AutoCompleteWidget(lookups.FavExpTypeLookup)
        self.fields['cell_type'].widget = \
            AutoCompleteWidget(lookups.FavExpCellTypeLookup)
        self.fields['assembly'].widget = \
            AutoCompleteWidget(lookups.FavExpAssemblyLookup)
        self.fields['target'].widget = \
            AutoCompleteWidget(lookups.FavExpTargetLookup)


class SimilarExperimentFilterForm(ExperimentFilterForm):

    def __init__(self, *args, **kwargs):
        pk = kwargs.pop('pk', None)
        print('form', pk)
        super().__init__(*args, **kwargs)

        # self.fields['search'].widget = \
        #     AutoCompleteWidget(lookups.SimExpSearchLookup)
        self.fields['name'].widget = \
            AutoCompleteWidget(lookups.SimExpNameLookup)
        # self.fields['description'].widget = \
        #     AutoCompleteWidget(lookups.SimExpDescriptionLookup)
        self.fields['data_type'].widget = \
            AutoCompleteWidget(lookups.SimExpDataTypeLookup)
        self.fields['cell_type'].widget = \
            AutoCompleteWidget(lookups.SimExpCellTypeLookup)
        self.fields['assembly'].widget = \
            AutoCompleteWidget(lookups.SimExpAssemblyLookup)
        self.fields['target'].widget = \
            AutoCompleteWidget(lookups.SimExpTargetLookup)

        for field in self.fields:
            if isinstance(self.fields[field].widget, AutoCompleteWidget):
                self.fields[field].widget.update_query_parameters(
                    {'pk': pk})


class BootstrapModelForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):

        super(BootstrapModelForm, self).__init__(*args, **kwargs)
        for field in iter(self.fields):
            self.fields[field].widget.attrs.update({
                'class': 'form-control',
            })
            if type(self.fields[field].widget) == forms.Textarea:
                self.fields[field].widget.attrs.update({
                    'rows': 2,
                })


class ExperimentForm(BootstrapModelForm):

    class Meta:
        model = models.Experiment
        fields = (
            'name', 'organism', 'experiment_type', 'cell_type', 'target',
            'description', 'use_default_color', 'color', 'public',
        )
        widgets = {
            'color': TextInput(attrs={'type': 'color'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['use_default_color'].widget.attrs.update({
            'class': 'checkbox',
        })
        self.fields['public'].widget.attrs.update({
            'class': 'checkbox',
        })


class DatasetForm(BootstrapModelForm):

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

        stranded_error_text = \
            'An ambiguous URL may not be included with stranded URLs.'

        def check_url(url):
            request = requests.head(url)
            if request.status_code >= 400:
                raise forms.ValidationError(
                    '{} is not a valid URL.'.format(url))

        if ambiguous_url:
            if plus_url or minus_url:
                raise forms.ValidationError(stranded_error_text)
            check_url(ambiguous_url)
        elif plus_url and minus_url:
            if ambiguous_url:
                raise forms.ValidationError(stranded_error_text)
            check_url(plus_url)
            check_url(minus_url)
        else:
            raise forms.ValidationError(
                'Either ambiguous or stranded URLs required.')


class UserForm(BootstrapModelForm):

    class Meta:
        model = User
        fields = (
            'username',
        )


class MyUserForm(BootstrapModelForm):

    class Meta:
        model = models.MyUser
        fields = (
            'public',
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['public'].widget.attrs.update({
            'class': 'checkbox',
        })
