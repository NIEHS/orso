import math
from collections import defaultdict

import numpy
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.views.generic \
    import View, TemplateView, DetailView, CreateView, UpdateView, DeleteView
from django.views.generic import ListView
from django.forms.models import inlineformset_factory
from django.db.models import F, Min, Q
from django.views.generic.base import ContextMixin
from django.contrib.auth.mixins import LoginRequiredMixin
from django.utils.decorators import available_attrs, method_decorator
from django.utils.cache import add_never_cache_headers
from django.urls import reverse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from functools import wraps

from . import models, forms


def get_name(request):
    if request.method == 'POST':
        form = forms.DatasetForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('/thanks/')
    else:
        form = forms.DatasetForm()

    return render(request, 'name.html', {'form': form})


def never_cache(view_func):
    """Add headers for no client-side caching."""
    @wraps(view_func, assigned=available_attrs(view_func))
    def _wrapped_view_func(request, *args, **kwargs):
        response = view_func(request, *args, **kwargs)
        add_never_cache_headers(response)
        if not response.has_header('Pragma'):
            response['Pragma'] = 'no-Cache'
        if not response.has_header('Cache-Control'):
            response['Cache-Control'] = 'no-Store, no-Cache'
        return response

    return _wrapped_view_func


@api_view()
def browser(request):
    chromosome = request.GET.get('chr')
    start = request.GET.get('start')
    end = request.GET.get('end')
    datasets = request.GET.get('datasets')
    return Response(models.Dataset.get_browser_view(
        chromosome, start, end, datasets))


class NeverCacheFormMixin:

    @method_decorator(never_cache)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)


class AddUserToFormMixin(LoginRequiredMixin):

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['owner'] = self.request.user
        return kwargs


class AddMyUserMixin(ContextMixin, LoginRequiredMixin):

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['login_user'] = models.MyUser.objects.get(
            user=self.request.user)
        return context


class ExperimentCreate(NeverCacheFormMixin, AddMyUserMixin, CreateView):
    model = models.Experiment
    form_class = forms.ExperimentForm
    DatasetFormSet = inlineformset_factory(
        models.Experiment, models.Dataset, form=forms.DatasetForm, extra=1)

    def get_success_url(self):
        return reverse('personal_experiments')

    def get(self, request, *args, **kwargs):
        self.object = None
        form = self.get_form(self.form_class)
        dataset_formset = self.DatasetFormSet(self.request.GET or None)
        return self.render_to_response(
            self.get_context_data(
                form=form,
                dataset_formset=dataset_formset))

    def post(self, request, *args, **kwargs):
        self.object = None
        form = self.get_form(self.form_class)
        dataset_formset = self.DatasetFormSet(self.request.POST or None)
        if form.is_valid() and dataset_formset.is_valid():
            return self.form_valid(form, dataset_formset)
        else:
            return self.form_invalid(form, dataset_formset)

    def form_valid(self, form, dataset_formset):
        context = super().get_context_data()
        login_user = context['login_user']

        self.object = form.save()
        self.object.owners.add(login_user)
        dataset_formset.instance = self.object
        dataset_formset.save()
        return HttpResponseRedirect(self.get_success_url())

    def form_invalid(self, form, dataset_formset):
        return self.render_to_response(
            self.get_context_data(
                form=form,
                dataset_formset=dataset_formset))


class ExperimentUpdate(NeverCacheFormMixin, AddMyUserMixin, UpdateView):
    model = models.Experiment
    form_class = forms.ExperimentForm
    DatasetFormSet = inlineformset_factory(
        models.Experiment, models.Dataset, form=forms.DatasetForm, extra=0)

    def get_success_url(self):
        return reverse('personal_experiments')

    def get_context_data(self, *args, **kwargs):
        context = super().get_context_data(*args, **kwargs)
        if self.request.POST:
            context['form'] = forms.ExperimentForm(
                self.request.POST, instance=self.object)
            context['datatset_formset'] = self.DatasetFormSet(
                self.request.POST, instance=self.object)
        else:
            context['form'] = forms.ExperimentForm(
                instance=self.object)
            context['dataset_formset'] = self.DatasetFormSet(
                instance=self.object)
        return context

    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        form = self.get_form(self.form_class)
        dataset_formset = self.DatasetFormSet(
            self.request.POST, instance=self.object)
        if form.is_valid() and dataset_formset.is_valid():
            return self.form_valid(form, dataset_formset)
        else:
            return self.form_invalid(form, dataset_formset)

    def form_valid(self, form, dataset_formset):
        self.object = form.save()
        dataset_formset.instance = self.object
        dataset_formset.save()
        return HttpResponseRedirect(self.get_success_url())

    def form_invalid(self, form, dataset_formset):
        return self.render_to_response(
            self.get_context_data(
                form=form,
                dataset_formset=dataset_formset))


class ExperimentDelete(NeverCacheFormMixin, DeleteView):
    model = models.Experiment
    form_class = forms.ExperimentForm

    def get_success_url(self):
        return reverse('personal_experiments')


class Index(View):
    def get(self, request, *args, **kwargs):
        return HttpResponse('Hello, world.')


class Home(TemplateView, AddMyUserMixin):
    template_name = 'network/home.html'


class Explore(TemplateView, AddMyUserMixin):
    template_name = 'network/explore.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        pca_lookup = dict()
        available_exp_types = dict()
        available_assemblies = dict()
        available_groups = []

        for pca in models.PCA.objects.all():
            exp_name = pca.experiment_type.name
            assembly_name = pca.locus_group.assembly.name
            group_name = pca.locus_group.group_type

            pca_lookup['{}:{}:{}'.format(
                assembly_name,
                exp_name,
                group_name,
            )] = pca.pk

            if assembly_name not in available_exp_types:
                available_exp_types[assembly_name] = []
            if exp_name not in available_exp_types[assembly_name]:
                available_exp_types[assembly_name].append(exp_name)

            if exp_name not in available_assemblies:
                available_assemblies[exp_name] = []
            if assembly_name not in available_assemblies[exp_name]:
                available_assemblies[exp_name].append(assembly_name)

            if group_name not in available_groups:
                available_groups.append(group_name)

        context['pca_lookup'] = pca_lookup
        context['available_experiment_types'] = available_exp_types
        context['available_assemblies'] = available_assemblies
        context['available_groups'] = available_groups

        return context


class Gene(DetailView, AddMyUserMixin):
    template_name = 'network/gene.html'
    model = models.Gene

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        gene = self.get_object()

        transcripts = models.Transcript.objects.filter(gene=gene)
        median_expressions = [t.get_median_expression() for t in transcripts]
        selected = gene.get_transcript_with_highest_expression()

        # Get transcript information
        context['transcripts'] = []
        for t, exp in zip(transcripts, median_expressions):
            context['transcripts'].append({
                'pk': t.pk,
                'name': t.name,
                'chromosome': t.chromosome,
                'strand': t.strand,
                'start': t.start,
                'end': t.end,
                'median_expression': exp,
                'selected': selected == t,
            })
        context['transcripts'].sort(key=lambda x: (
            -x['selected'], -x['median_expression']))

        # Get expression information
        datasets = models.Dataset.objects.filter(
            assembly__annotation=gene.annotation,
            experiment__experiment_type__name='RNA-seq',
        )
        expression = defaultdict(list)
        for dataset in datasets:
            expression[dataset.experiment.cell_type].append(
                models.TranscriptIntersection.objects.get(
                    dataset=dataset,
                    transcript=selected,
                ).normalized_coding_value
            )
        context['expression'] = []
        for cell_type, values in sorted(
                expression.items(), key=lambda x: numpy.median(x[1])):
            context['expression'].append({
                'cell_type': cell_type,
                'median': numpy.median(values),
                'min': min(values),
                'max': max(values),
            })

        return context


class Transcript(DetailView, AddMyUserMixin):
    template_name = 'network/transcript.html'
    model = models.Transcript

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        transcript = self.get_object()
        context['median_expression'] = transcript.get_median_expression()

        other_transcripts = models.Transcript.objects.filter(
            gene=transcript.gene).exclude(pk=transcript.pk)
        median_expressions = [
            t.get_median_expression() for t in other_transcripts]

        # Get transcript information
        context['other_transcripts'] = []
        for t, exp in zip(other_transcripts, median_expressions):
            context['other_transcripts'].append({
                'pk': t.pk,
                'name': t.name,
                'chromosome': t.chromosome,
                'strand': t.strand,
                'start': t.start,
                'end': t.end,
                'median_expression': exp,
            })
        context['other_transcripts'].sort(
            key=lambda x: -x['median_expression'])

        # Get expression information
        datasets = models.Dataset.objects.filter(
            assembly__annotation=transcript.gene.annotation,
            experiment__experiment_type__name='RNA-seq',
        )
        expression = defaultdict(list)
        for dataset in datasets:
            expression[dataset.experiment.cell_type].append(
                models.TranscriptIntersection.objects.get(
                    dataset=dataset,
                    transcript=transcript,
                ).normalized_coding_value
            )
        context['expression'] = []
        for cell_type, values in sorted(
                expression.items(), key=lambda x: numpy.median(x[1])):
            context['expression'].append({
                'cell_type': cell_type,
                'median': numpy.median(values),
                'min': min(values),
                'max': max(values),
            })

        return context


class Experiment(DetailView, AddMyUserMixin):
    template_name = 'network/experiment.html'
    model = models.Experiment

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        login_user = context['login_user']
        exp = self.get_object()

        context['selectable'] = dict()
        context['selectable']['personal'] = \
            login_user.get_personal_experiment_ids()
        context['selectable']['favorite'] = \
            login_user.get_favorite_experiment_ids()

        context['display_data'] = exp.get_display_data(context['login_user'])
        context['plot_data'] = exp.get_average_metaplots()

        context['datasets'] = models.Dataset.objects.filter(
            experiment=self.get_object())

        gene_values = []
        gene_set = set()

        for intersection in models.ExperimentIntersection.objects.filter(
            experiment=self.get_object(),
            locus__group__group_type='genebody'
        ).order_by('-average_value'):

            transcript = intersection.locus.from_genebody.get()
            if all([
                transcript.end - transcript.start >= 200,
                transcript.gene.name not in gene_set,
            ]):
                gene_values.append([
                    transcript.gene.name,
                    intersection.average_value,
                ])
                gene_set.add(transcript.gene.name)
            if len(gene_values) >= 20:
                break

        context['gene_values'] = gene_values

        return context


class Dataset(DetailView, AddMyUserMixin):
    template_name = 'network/dataset.html'
    model = models.Dataset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context['plot_data'] = self.get_object().get_metaplots()

        gene_values = []
        gene_set = set()

        for intersection in models.DatasetIntersection.objects.filter(
            dataset=self.get_object(),
            locus__group__group_type='genebody'
        ).order_by('-normalized_value'):

            transcript = intersection.locus.from_genebody.get()
            if all([
                transcript.end - transcript.start >= 200,
                transcript.gene.name not in gene_set,
            ]):
                gene_values.append([
                    transcript.gene.name,
                    intersection.normalized_value,
                ])
                gene_set.add(transcript.gene.name)
            if len(gene_values) >= 20:
                break

        context['gene_values'] = gene_values

        return context


class MyUser(DetailView, AddMyUserMixin):
    template_name = 'network/user.html'
    model = models.MyUser

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        my_user = self.get_object()
        login_user = context['login_user']

        latest = (models.Experiment.objects
                        .filter(owners__in=[my_user])
                        .latest())
        context.update(my_user.get_display_data(login_user))
        context['latest_dataset'] = latest.get_display_data(login_user)

        return context


class ExperimentList(AddMyUserMixin, ListView):
    def dispatch(self, request, *args, **kwargs):
        self.my_user = models.MyUser.objects.get(user=self.request.user)
        return super().dispatch(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        try:
            pk = self.kwargs['pk']
            if self.request.GET:
                self.form = self.form_class(
                    self.request.GET, pk=pk,
                )
            else:
                self.form = self.form_class(pk=pk)
        except:
            if self.request.GET:
                self.form = self.form_class(
                    self.request.GET
                )
            else:
                self.form = self.form_class()
        return super().get(request, *args, **kwargs)

    def get_paginate_by(self, qs):
        val = 10
        try:
            val = int(self.request.GET.get('paginate_by', val))
        except ValueError:
            pass
        return val

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['experiment_counts'] = self.my_user.get_experiment_counts()
        context['form'] = self.form
        context['search_field'] = self.form['search']
        context['other_fields'] = []
        for field in self.form:
            if field.name != 'search':
                context['other_fields'].append(field)
        return context


class PersonalExperiments(ExperimentList):
    model = models.Experiment
    template_name = 'network/personal_experiments.html'
    form_class = forms.PersonalExperimentFilterForm

    def get_queryset(self):

        query = Q(owners__in=[self.my_user])

        if self.form.is_valid():
            query &= self.form.get_query()

        qs = self.model.objects.filter(query)

        for obj in qs:
            obj.plot_data = obj.get_average_metaplots()
            obj.meta_data = obj.get_metadata(self.my_user)
            obj.urls = obj.get_urls()

        return qs


class FavoriteExperiments(ExperimentList):
    model = models.Experiment
    template_name = 'network/favorite_experiments.html'
    form_class = forms.FavoriteExperimentFilterForm

    def get_queryset(self):

        query = Q(experimentfavorite__owner=self.my_user)

        if self.form.is_valid():
            query &= self.form.get_query()

        qs = self.model.objects.filter(query)

        for obj in qs:
            obj.plot_data = obj.get_average_metaplots()
            obj.meta_data = obj.get_metadata(self.my_user)
            obj.urls = obj.get_urls()

        return qs


# TODO: Have RecommendedExperiments and SimilarExperiments inherit from the
# same class
class RecommendedExperiments(ExperimentList):
    model = models.Experiment
    template_name = 'network/recommended_experiments.html'
    form_class = forms.RecommendedExperimentFilterForm

    def get_queryset(self):

        user_experiments = models.Experiment.objects.filter(
            owners=self.my_user)

        order = self.form_class.order_choices[0][0]
        if self.form.is_valid():
            order = self.form.get_order()

        # Get recommended experiments
        if order == 'correlation_rank':
            _query = Q(network_experimentdatadistance_first__experiment_2__in=user_experiments)  # noqa
            _query &= ~Q(network_experimentdatadistance_second__experiment_1__in=user_experiments)  # noqa
        elif order == 'metadata_rank':
            _query = Q(network_experimentmetadatadistance_first__experiment_2__in=user_experiments)  # noqa
            _query &= ~Q(network_experimentmetadatadistance_second__experiment_1__in=user_experiments)  # noqa

        if self.form.is_valid():
            _query &= self.form.get_query()

        if order == 'correlation_rank':
            agg = 'network_experimentdatadistance_first__distance'
        elif order == 'metadata_rank':
            agg = 'network_experimentmetadatadistance_first__distance'
        qs = (self.model
                  .objects
                  .filter(_query)
                  .annotate(min_distance=Min(agg))
                  .order_by('min_distance'))

        for obj in qs:
            obj.plot_data = obj.get_average_metaplots()
            obj.meta_data = obj.get_metadata(self.my_user)
            obj.urls = obj.get_urls()

        return qs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        user_experiments = models.Experiment.objects.filter(
            owners=self.my_user)

        order = self.form_class.order_choices[0][0]
        if self.form.is_valid():
            order = self.form.get_order()

        # Get score distribution
        _query = Q(experiment_2__in=user_experiments)
        _query &= ~Q(experiment_1__in=user_experiments)

        if order == 'correlation_rank':
            all_distances = models.ExperimentDataDistance.objects.filter(
                _query).values_list('distance', flat=True)
        elif order == 'metadata_rank':
            all_distances = models.ExperimentMetadataDistance.objects.filter(
                _query).values_list('distance', flat=True)

        context['all_distances'] = list(all_distances)

        return context


class SimilarExperiments(ExperimentList):
    model = models.Experiment
    template_name = 'network/similar_experiments.html'
    form_class = forms.SimilarExperimentFilterForm

    def get_queryset(self):
        exp = models.Experiment.objects.get(pk=self.kwargs['pk'])

        order = self.form_class.order_choices[0][0]
        if self.form.is_valid():
            order = self.form.get_order()

        # Get recommended experiments
        if order == 'correlation_rank':
            _query = Q(network_experimentdatadistance_first__experiment_2=exp)  # noqa
            _query &= ~Q(network_experimentdatadistance_second__experiment_1=exp)  # noqa
        elif order == 'metadata_rank':
            _query = Q(network_experimentmetadatadistance_first__experiment_2=exp)  # noqa
            _query &= ~Q(network_experimentmetadatadistance_second__experiment_1__in=exp)  # noqa

        if self.form.is_valid():
            _query &= self.form.get_query()

        if order == 'correlation_rank':
            agg = 'network_experimentdatadistance_first__distance'
        elif order == 'metadata_rank':
            agg = 'network_experimentmetadatadistance_first__distance'
        qs = (self.model
                  .objects
                  .filter(_query)
                  .annotate(min_distance=Min(agg))
                  .order_by('min_distance'))

        for obj in qs:
            obj.plot_data = obj.get_average_metaplots()
            obj.meta_data = obj.get_metadata(self.my_user)
            obj.urls = obj.get_urls()

        return qs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        exp = models.Experiment.objects.get(pk=self.kwargs['pk'])

        order = self.form_class.order_choices[0][0]
        if self.form.is_valid():
            order = self.form.get_order()

        # Get score distribution
        _query = Q(experiment_2=exp)
        _query &= ~Q(experiment_1=exp)

        if order == 'correlation_rank':
            all_distances = models.ExperimentDataDistance.objects.filter(
                _query).values_list('distance', flat=True)
        elif order == 'metadata_rank':
            all_distances = models.ExperimentMetadataDistance.objects.filter(
                _query).values_list('distance', flat=True)

        context['all_distances'] = list(all_distances)

        return context


class PCA(AddMyUserMixin, DetailView):
    template_name = 'network/pca.html'
    model = models.PCA

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        exp_urls = dict()
        for exp in models.Experiment.objects.all():
            exp_urls[exp.pk] = exp.get_absolute_url()
        context['exp_urls'] = exp_urls

        return context


class FavoriteUsers(TemplateView, AddMyUserMixin):
    template_name = 'network/favorite_users.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        my_user = models.MyUser.objects.get(user=self.request.user)

        users = []
        for fav in models.UserFavorite.objects.filter(owner=my_user):
            users.append(fav.favorite.get_display_data(my_user))

        context['users'] = users
        context['user_counts'] = my_user.get_user_counts()

        return context


class RecommendedUsers(TemplateView, AddMyUserMixin):
    template_name = 'network/recommended_users.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        my_user = models.MyUser.objects.get(user=self.request.user)

        users = []
        for rec in models.UserRecommendation.objects.filter(owner=my_user):
            users.append(rec.recommended.get_display_data(my_user))

        context['users'] = users
        context['user_counts'] = my_user.get_user_counts()

        return context


class DatasetComparison(TemplateView, AddMyUserMixin):
    template_name = 'network/dataset_comparison.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        ds_x = models.Dataset.objects.get(pk=kwargs['x'])
        ds_y = models.Dataset.objects.get(pk=kwargs['y'])

        if (ds_x.assembly == ds_y.assembly) and \
                (ds_x.experiment.experiment_type ==
                 ds_y.experiment.experiment_type):

            genes = models.Gene.objects.filter(
                annotation__assembly=ds_x.assembly,
                selected_transcript__isnull=False,
            ).order_by('pk').values_list('name', flat=True)

            intersections_x = (
                models.TranscriptIntersection
                      .objects
                      .annotate(
                          size=F('transcript__end') - F('transcript__start'))
                      .filter(
                          dataset=ds_x,
                          transcript__selecting__isnull=False,
                          size__gte=200)
                      .order_by('transcript__selecting__pk')
                      .values_list('normalized_coding_value', flat=True))
            intersections_y = (
                models.TranscriptIntersection
                      .objects
                      .annotate(
                          size=F('transcript__end') - F('transcript__start'))
                      .filter(
                          dataset=ds_y,
                          transcript__selecting__isnull=False,
                          size__gte=200)
                      .order_by('transcript__selecting__pk')
                      .values_list('normalized_coding_value', flat=True))

            context['scatter'] = \
                [[gene, x, y] for gene, x, y in
                 zip(genes, intersections_x, intersections_y)
                 if (x, y) != (0, 0)]

            log_change = dict()
            for gene, int_x, int_y in zip(
                    genes, intersections_x, intersections_y):
                log_change[gene] = math.log2((int_x + 1) / (int_y + 1))
            log_change = \
                sorted(log_change.items(), key=lambda x: -abs(x[1]))[:100]

            context['log_change'] = [list(x) for x in sorted(
                log_change, key=lambda x: -x[1])]

        context['dataset_x'] = ds_x
        context['dataset_y'] = ds_y

        return context


class ExperimentComparison(TemplateView, AddMyUserMixin):
    template_name = 'network/experiment_comparison.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        exp_x = models.Experiment.objects.get(pk=kwargs['x'])
        exp_y = models.Experiment.objects.get(pk=kwargs['y'])

        exp_x_assemblies = models.Assembly.objects.filter(
            dataset__experiment=exp_x)
        exp_y_assemblies = models.Assembly.objects.filter(
            dataset__experiment=exp_y)

        context['experiment_x_assemblies'] = ', '.join(list(set(
            exp_x_assemblies.values_list('name', flat=True))))
        context['experiment_y_assemblies'] = ', '.join(list(set(
            exp_y_assemblies.values_list('name', flat=True))))

        context['experiment_x'] = exp_x
        context['experiment_y'] = exp_y

        return context


class TestSmallDataView(TemplateView, AddMyUserMixin):
    template_name = 'network/test_small_data_view.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        my_user = models.MyUser.objects.get(user=self.request.user)

        datasets = []
        for ds in models.Dataset.objects.filter(owners__in=[my_user]):
            datasets.append(ds.get_display_data(my_user))

        context['datasets'] = datasets
        context['dataset_counts'] = my_user.get_dataset_counts()

        return context


class TestSmallUserView(TemplateView, AddMyUserMixin):
    template_name = 'network/test_small_user_view.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        my_user = models.MyUser.objects.get(user=self.request.user)

        users = []
        for fav in models.UserFavorite.objects.filter(owner=my_user):
            users.append(fav.favorite.get_display_data(my_user))

        context['users'] = users
        context['user_counts'] = my_user.get_user_counts()

        return context
