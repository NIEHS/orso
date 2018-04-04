import math
from collections import defaultdict

import numpy
from django.core.exceptions import PermissionDenied
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views.generic \
    import View, TemplateView, DetailView, CreateView, UpdateView, DeleteView
from django.views.generic import ListView
from django.forms.models import inlineformset_factory
from django.db.models import Q
from django.views.generic.base import ContextMixin
from django.views.generic.detail import SingleObjectMixin
from django.contrib.auth.mixins import LoginRequiredMixin
from django.utils.decorators import available_attrs, method_decorator
from django.utils.cache import add_never_cache_headers
from django.urls import reverse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from functools import wraps

from . import models, forms
from analysis.utils import generate_intersection_df
from network.tasks.process_experiments import process_experiment


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


class AddMyUserMixin(ContextMixin):

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if self.request.user.is_authenticated():
            context['login_user'] = models.MyUser.objects.get(
                user=self.request.user)
        else:
            context['login_user'] = None
        return context


class ExperimentCreate(LoginRequiredMixin, NeverCacheFormMixin, AddMyUserMixin,
                       CreateView):
    template_name = 'experiments/create.html'
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

        process_experiment(self.object.pk)

        models.Activity.objects.create(
            user=login_user,
            created_experiment=self.object,
        )

        return HttpResponseRedirect(self.get_success_url())

    def form_invalid(self, form, dataset_formset):
        return self.render_to_response(
            self.get_context_data(
                form=form,
                dataset_formset=dataset_formset))


class ExperimentUpdate(LoginRequiredMixin, NeverCacheFormMixin, AddMyUserMixin,
                       UpdateView):
    template_name = 'experiments/update.html'
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


class ExperimentDelete(LoginRequiredMixin, NeverCacheFormMixin, DeleteView):
    model = models.Experiment
    form_class = forms.ExperimentForm
    template_name = 'experiments/delete.html'

    def get_success_url(self):
        return reverse('personal_experiments')


class MyUserUpdate(AddMyUserMixin, LoginRequiredMixin, NeverCacheFormMixin,
                   UpdateView):
    model = models.MyUser
    template_name = 'users/update.html'
    fields = ('public',)

    def get_success_url(self):
        return reverse('user', args=[self.object.pk])


class MyUserDelete(AddMyUserMixin, LoginRequiredMixin, NeverCacheFormMixin,
                   DeleteView):
    model = models.MyUser
    template_name = 'users/delete.html'

    def get_success_url(self):
        return reverse('home')


class Index(View):
    def get(self, request, *args, **kwargs):
        return HttpResponse('Hello, world.')


class Home(AddMyUserMixin, ListView):
    template_name = 'network/home.html'
    model = models.Activity

    def dispatch(self, request, *args, **kwargs):
        if self.request.user.is_authenticated():
            self.my_user = models.MyUser.objects.get(user=self.request.user)
        else:
            self.my_user = None
        return super().dispatch(request, *args, **kwargs)

    def get_queryset(self, **kwargs):

        query = Q()
        # Followed user performs an action
        query |= Q(user__following__followed=self.my_user)
        # Another user followed the requesting user
        query |= Q(followed_user=self.my_user)

        return self.model.objects.filter(query).order_by('-last_updated')


class ExplorePCA(TemplateView, AddMyUserMixin):
    template_name = 'explore/pca.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        user_dict = defaultdict(list)

        if context['login_user']:

            for trans in models.PCATransformedValues.objects.filter(
                    dataset__experiment__owners=context['login_user']):
                user_dict[trans.pca.pk].append({
                    'colors': {'Default': '#A9A9A9'},
                    'dataset_name': trans.dataset.name,
                    'dataset_pk': trans.dataset.pk,
                    'experiment_cell_type': trans.dataset.experiment.cell_type,
                    'experiment_name': trans.dataset.experiment.name,
                    'experiment_pk': trans.dataset.experiment.pk,
                    'experiment_target': trans.dataset.experiment.target,
                    'transformed_values': trans.transformed_values,
                })

        context['user_data'] = dict(user_dict)

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


class ExploreOverview(TemplateView, AddMyUserMixin):
    template_name = 'explore/overview.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context['experiment_number'] = models.Experiment.objects.all().count()
        context['dataset_number'] = models.Dataset.objects.all().count()
        context['encode_dataset_number'] = models.Dataset.objects.filter(
            experiment__project__name='ENCODE').count()

        context['user_number'] = models.MyUser.objects.all().count()
        context['dataset_from_users_number'] = models.Dataset.objects.exclude(
            experiment__owners=None).count()
        context['experiment_from_users_number'] = \
            models.Experiment.objects.exclude(owners=None).count()

        return context


class ExploreRecommendations(TemplateView, AddMyUserMixin):
    template_name = 'explore/recommendations.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        paired_data = dict()
        for data_dist in models.DatasetDataDistance.objects.all():
            try:
                metadata_dist = models.DatasetMetadataDistance.objects.get(
                    dataset_1=data_dist.dataset_1,
                    dataset_2=data_dist.dataset_2,
                )
            except models.DatasetMetadataDistance.DoesNotExist:
                pass
            else:
                exp_type = data_dist.dataset_1.experiment.experiment_type.name

                if exp_type not in paired_data:
                    paired_data[exp_type] = []

                paired_data[exp_type].append([
                    data_dist.dataset_1.pk,
                    data_dist.dataset_1.name,
                    data_dist.dataset_2.pk,
                    data_dist.dataset_2.name,
                    data_dist.distance,
                    metadata_dist.distance,
                ])

        quartiled_data = dict()
        for exp_type, _data in paired_data.items():
            _sorted = sorted(_data, key=lambda x: (-x[5], -x[4]))

            if len(_sorted) >= 4:
                quartiled = [arr.tolist() for arr in numpy.array_split(
                    _sorted, 4)]
            else:
                quartiled = [arr.tolist() for arr in numpy.array_split(
                    _sorted, len(_sorted))]

            quartiled_data[exp_type] = []
            for _list in quartiled:
                quartiled_data[exp_type].append([x[4] for x in _list])

        context['data'] = {
            'paired_data': paired_data,
            'quartiled_data': quartiled_data,
        }

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


class CheckPublicMixin(SingleObjectMixin):

    def get_object(self, **kwargs):
        obj = super().get_object(**kwargs)

        if not obj.is_public():
            if self.request.user.is_authenticated():
                if not obj.is_owned(self.request.user):
                    raise PermissionDenied
            else:
                raise PermissionDenied

        return obj


class CheckPublicExperimentMixin(CheckPublicMixin):
    model = models.Experiment


class CheckPublicDatasetMixin(CheckPublicMixin):
    model = models.Dataset


class CheckPublicMyUserMixin(CheckPublicMixin):
    model = models.MyUser


class Experiment(CheckPublicExperimentMixin, AddMyUserMixin, DetailView):
    template_name = 'experiments/experiment.html'
    model = models.Experiment

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        exp = self.get_object()

        data_lookup = defaultdict(dict)
        for mp in models.MetaPlot.objects.filter(
                dataset__experiment=exp):
            key = '{}:{}'.format(mp.dataset.name, mp.locus_group.group_type)
            data_lookup[key].update({'metaplot': mp.pk})
        for fv in models.FeatureValues.objects.filter(
                dataset__experiment=exp):
            key = '{}:{}'.format(fv.dataset.name, fv.locus_group.group_type)
            data_lookup[key].update({'feature_values': fv.pk})

        context['data_lookup'] = dict(data_lookup)
        context['datasets'] = models.Dataset.objects.filter(experiment=exp)
        context['is_favorite'] = models.Favorite.objects.filter(
            experiment=exp,
            user=context['login_user'],
        ).exists()
        context['owned'] = \
            exp.owners.filter(pk=context['login_user'].pk).exists()

        return context


class Dataset(CheckPublicDatasetMixin, DetailView, AddMyUserMixin):
    template_name = 'datasets/dataset.html'
    model = models.Dataset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        dataset = self.get_object()

        data_lookup = defaultdict(dict)
        for mp in models.MetaPlot.objects.filter(
                dataset=dataset):
            data_lookup[mp.locus_group.group_type].update(
                {'metaplot': mp.pk})
        for fv in models.FeatureValues.objects.filter(
                dataset=dataset):
            data_lookup[fv.locus_group.group_type].update(
                {'feature_values': fv.pk})

        context['data_lookup'] = dict(data_lookup)
        context['is_favorite'] = models.Favorite.objects.filter(
            experiment=dataset.experiment,
            user=context['login_user'],
        ).exists()
        context['owned'] = dataset.experiment.owners.filter(
            pk=context['login_user'].pk).exists()

        return context


class MyUser(CheckPublicMyUserMixin, DetailView, AddMyUserMixin):
    template_name = 'users/user.html'
    model = models.MyUser

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        my_user = self.get_object()
        login_user = context['login_user']
        context.update(my_user.get_display_data(login_user))
        return context


class ExperimentList(AddMyUserMixin, ListView):

    display_experiment_navbar = True

    def dispatch(self, request, *args, **kwargs):
        if self.request.user.is_authenticated():
            self.my_user = models.MyUser.objects.get(user=self.request.user)
        else:
            self.my_user = None
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

    def get_page_objs(self, qs):
        paginator = Paginator(qs, self.get_paginate_by(qs))
        page = self.request.GET.get('page')

        try:
            current_objects = paginator.page(page)
        except PageNotAnInteger:
            current_objects = paginator.page(1)
        except EmptyPage:
            current_objects = paginator.page(paginator.num_pages)

        return current_objects

    def get_queryset(self, base_query):

        query = base_query
        query &= Q(dataset__processed=True)  # Only returned processed exps

        if self.form.is_valid():
            query &= self.form.get_query()

        qs = models.Experiment.objects.filter(query).distinct().order_by('pk')

        paginator = Paginator(qs, self.get_paginate_by(qs))
        page = self.request.GET.get('page')

        try:
            current_objects = paginator.page(page)
        except PageNotAnInteger:
            current_objects = paginator.page(1)
        except EmptyPage:
            current_objects = paginator.page(paginator.num_pages)

        for obj in qs:
            if obj in current_objects:

                obj.plot_data = obj.get_average_metaplots()
                obj.meta_data = obj.get_metadata(self.my_user)
                obj.urls = obj.get_urls()

        return qs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['display_experiment_navbar'] = self.display_experiment_navbar
        # context['experiment_counts'] = self.my_user.get_experiment_counts()
        context['experiment_counts'] = 0
        context['form'] = self.form
        context['search_field'] = self.form['search']
        context['other_fields'] = []
        for field in self.form:
            if field.name != 'search':
                context['other_fields'].append(field)
        return context


class AllExperiments(ExperimentList):
    template_name = 'experiments/all_experiments.html'
    form_class = forms.AllExperimentFilterForm

    def get_queryset(self):
        base_query = Q(public=True)
        return super().get_queryset(base_query)


class PersonalExperiments(LoginRequiredMixin, ExperimentList):
    template_name = 'experiments/personal_experiments.html'
    form_class = forms.PersonalExperimentFilterForm

    def get_queryset(self):
        base_query = Q(owners__in=[self.my_user])
        return super().get_queryset(base_query)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['processing'] = models.Experiment.objects.filter(
            dataset__processed=False)
        return context


class FavoriteExperiments(LoginRequiredMixin, ExperimentList):
    template_name = 'experiments/favorite_experiments.html'
    form_class = forms.FavoriteExperimentFilterForm

    def get_queryset(self):
        base_query = Q(favorite__user=self.my_user)
        return super().get_queryset(base_query)


class UserExperiments(ExperimentList):
    template_name = 'experiments/user_experiments.html'
    form_class = forms.AllExperimentFilterForm
    display_experiment_navbar = False

    def get(self, request, *args, **kwargs):
        self.target_user = models.MyUser.objects.get(pk=self.kwargs['pk'])
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        base_query = Q(owners__in=[self.target_user])
        return super().get_queryset(base_query)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['target_user'] = self.target_user
        return context


# TODO: Have RecommendedExperiments and SimilarExperiments inherit from the
# same class
class RecommendedExperiments(LoginRequiredMixin, ExperimentList):
    model = models.Experiment
    template_name = 'experiments/recommended_experiments.html'
    form_class = forms.RecommendedExperimentFilterForm

    def get_queryset(self):

        query = (
            ~Q(owners=self.my_user) & ~Q(favorite__user=self.my_user) &
            Q(recommended_experiment__user=self.my_user)
        )

        if self.form.is_valid():
            query &= self.form.get_query()

        qs = self.model.objects.filter(query).distinct()

        paginator = Paginator(qs, self.get_paginate_by(qs))
        page = self.request.GET.get('page')

        try:
            current_objects = paginator.page(page)
        except PageNotAnInteger:
            current_objects = paginator.page(1)
        except EmptyPage:
            current_objects = paginator.page(paginator.num_pages)

        for obj in qs:
            if obj in current_objects:

                obj.plot_data = obj.get_average_metaplots()
                obj.meta_data = obj.get_metadata(self.my_user)
                obj.urls = obj.get_urls()
                obj.recommendation_tags = \
                    obj.get_recommendation_tags(self.my_user)

        return qs


class SimilarExperiments(ExperimentList):
    model = models.Experiment
    template_name = 'experiments/similar_experiments.html'
    form_class = forms.SimilarExperimentFilterForm

    def get_queryset(self):

        exp = models.Experiment.objects.get(pk=self.kwargs['pk'])

        query = Q(public=True) & (
            Q(sim_experiment_1__experiment_2=exp) |
            Q(sim_experiment_2__experiment_1=exp)
        )

        if self.form.is_valid():
            query &= self.form.get_query()

        qs = self.model.objects.filter(query).distinct()

        paginator = Paginator(qs, self.get_paginate_by(qs))
        page = self.request.GET.get('page')

        try:
            current_objects = paginator.page(page)
        except PageNotAnInteger:
            current_objects = paginator.page(1)
        except EmptyPage:
            current_objects = paginator.page(paginator.num_pages)

        for obj in qs:
            if obj in current_objects:

                obj.plot_data = obj.get_average_metaplots()
                obj.meta_data = obj.get_metadata(self.my_user)
                obj.urls = obj.get_urls()

        return qs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context['ref_experiment'] = \
            models.Experiment.objects.get(pk=self.kwargs['pk'])

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


class UserList(AddMyUserMixin, ListView):
    model = models.MyUser
    form_class = forms.UserFilterForm
    display_user_navbar = False

    def dispatch(self, request, *args, **kwargs):
        if self.request.user.is_authenticated():
            self.my_user = models.MyUser.objects.get(user=self.request.user)
        else:
            self.my_user = None
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

    def get_page_objs(self, qs):
        paginator = Paginator(qs, self.get_paginate_by(qs))
        page = self.request.GET.get('page')

        try:
            current_objects = paginator.page(page)
        except PageNotAnInteger:
            current_objects = paginator.page(1)
        except EmptyPage:
            current_objects = paginator.page(paginator.num_pages)

        return current_objects

    def get_queryset(self, base_query):

        query = base_query

        if self.form.is_valid():
            query &= self.form.get_query()

        qs = self.model.objects.filter(query).distinct().order_by('pk')

        paginator = Paginator(qs, self.get_paginate_by(qs))
        page = self.request.GET.get('page')

        try:
            current_objects = paginator.page(page)
        except PageNotAnInteger:
            current_objects = paginator.page(1)
        except EmptyPage:
            current_objects = paginator.page(paginator.num_pages)

        for obj in qs:
            if obj in current_objects:

                display_data = obj.get_display_data(self.my_user)

                obj.plot_data = display_data['plot_data']
                obj.meta_data = display_data['meta_data']
                obj.urls = display_data['urls']

        return qs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context['display_user_navbar'] = self.display_user_navbar
        context['form'] = self.form
        context['search_field'] = self.form['search']
        context['other_fields'] = []
        for field in self.form:
            if field.name != 'search':
                context['other_fields'].append(field)

        return context


class AllUsers(UserList):
    template_name = 'users/all_users.html'

    def get_queryset(self):
        base_query = ~Q(pk=self.my_user.pk) & Q(public=True)
        return super().get_queryset(base_query)


class Followed(LoginRequiredMixin, UserList):
    template_name = 'users/all_users.html'

    def get_queryset(self):
        base_query = Q(followed__following=self.my_user)
        return super().get_queryset(base_query)


class Followers(LoginRequiredMixin, UserList):
    template_name = 'users/all_users.html'

    def get_queryset(self):
        base_query = Q(following__followed=self.my_user)
        return super().get_queryset(base_query)


class UserFollowed(UserList):
    template_name = 'users/user_followed.html'
    display_user_navbar = False

    def get(self, request, *args, **kwargs):
        self.target_user = models.MyUser.objects.get(pk=self.kwargs['pk'])
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        base_query = Q(followed__following=self.target_user)
        return super().get_queryset(base_query)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['target_user'] = self.target_user
        return context


class UserFollowers(UserList):
    template_name = 'users/user_followers.html'
    display_user_navbar = False

    def get(self, request, *args, **kwargs):
        self.target_user = models.MyUser.objects.get(pk=self.kwargs['pk'])
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        base_query = Q(following__followed=self.target_user)
        return super().get_queryset(base_query)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['target_user'] = self.target_user
        return context


class FavoriteUsers(LoginRequiredMixin, TemplateView, AddMyUserMixin):
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

        if (ds_x.assembly != ds_y.assembly and
                ds_x.experiment.experiment_type !=
                ds_y.experiment.experiment_type):
            raise ValueError('Assembly and experiment type do not match.')

        locus_group = models.LocusGroup.objects.get(
            assembly=ds_x.assembly,
            group_type=ds_x.experiment.experiment_type.relevant_regions,
        )
        loci = models.Locus.objects.filter(
            group=locus_group,
            pca__experiment_type=ds_x.experiment.experiment_type
        )

        df = generate_intersection_df(
            locus_group,
            ds_x.experiment.experiment_type,
            datasets=[ds_x, ds_y],
            loci=loci,
        )

        context['dataset_x'] = ds_x
        context['dataset_y'] = ds_y

        context['scatter'] = [[pk, x, y] for pk, x, y in zip(
            list(df.index),
            list(df[ds_x.pk].values),
            list(df[ds_y.pk].values),
        ) if (x, y) != (0, 0)]

        log_fold_changes = \
            [[pk, math.log2((x + 1) / (y + 1))] for pk, x, y in zip(
                list(df.index),
                list(df[ds_x.pk].values),
                list(df[ds_y.pk].values),
            )]
        log_fold_changes = \
            sorted(log_fold_changes, key=lambda x: -abs(x[1]))[:20]
        log_fold_changes = [[
            models.Locus.objects.get(pk=pk).get_name(),
            lfc,
        ] for pk, lfc in log_fold_changes]

        context['log_change'] = log_fold_changes

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
