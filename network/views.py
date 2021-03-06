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
from django.contrib.auth.models import User
from django.utils.decorators import available_attrs, method_decorator
from django.utils.cache import add_never_cache_headers
from django.urls import reverse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from functools import wraps

from . import models, forms
from network.tasks.analysis.utils import generate_intersection_df
from network.tasks.processing import process_experiment
from network.tasks.utils import get_target_color, get_cell_type_color


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


@api_view()
def network(request):
    org_pk = request.GET.get('organism')
    exp_type = request.GET.get('exp-type')

    organism = models.Organism.objects.get(pk=org_pk)
    exp_type = models.ExperimentType.objects.get(pk=exp_type)

    return Response(organism.get_network(exp_type))


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

    def get_object(self, **kwargs):
        obj = super().get_object(**kwargs)

        if self.request.user.is_authenticated():
            my_user = models.MyUser.objects.get(user=self.request.user)
            if not obj.owners.filter(pk=my_user.pk).exists():
                raise PermissionDenied

        return obj

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


class ExperimentDelete(AddMyUserMixin, LoginRequiredMixin, NeverCacheFormMixin,
                       DeleteView):
    model = models.Experiment
    form_class = forms.ExperimentForm
    template_name = 'experiments/delete.html'

    def get_object(self, **kwargs):
        obj = super().get_object(**kwargs)

        if self.request.user.is_authenticated():
            my_user = models.MyUser.objects.get(user=self.request.user)
            if not obj.owners.filter(pk=my_user.pk).exists():
                raise PermissionDenied

        return obj

    def get_success_url(self):
        return reverse('personal_experiments')


class UserUpdate(AddMyUserMixin, LoginRequiredMixin, NeverCacheFormMixin,
                 UpdateView):
    model = User
    template_name = 'users/update.html'
    form_class = forms.UserForm

    user_form_class = form_class
    my_user_form_class = forms.MyUserForm

    def get_success_url(self):
        return reverse('user', args=[self.object.pk])

    def get_object(self, **kwargs):
        obj = super().get_object(**kwargs)

        if self.request.user.is_authenticated():
            if obj != self.request.user:
                raise PermissionDenied

        return obj

    def get_context_data(self, *args, **kwargs):
        context = super().get_context_data(*args, **kwargs)
        my_user = models.MyUser.objects.get(user=self.object)

        if self.request.POST:
            context['user_form'] = self.user_form_class(
                self.request.POST, instance=self.object)
            context['my_user_form'] = self.my_user_form_class(
                self.request.POST, instance=my_user)
        else:
            context['user_form'] = self.user_form_class(
                instance=self.object)
            context['my_user_form'] = self.my_user_form_class(
                instance=my_user)
        return context

    def post(self, request, *args, **kwargs):
        self.object = self.get_object()

        user = self.object
        my_user = models.MyUser.objects.get(user=user)

        user_form = self.user_form_class(
            self.request.POST, instance=user)
        my_user_form = self.my_user_form_class(
            self.request.POST, instance=my_user)

        if user_form.is_valid() and my_user_form.is_valid():
            return self.form_valid(user_form, my_user_form)
        else:
            return self.form_invalid(user_form, my_user_form)

    def form_valid(self, user_form, my_user_form):
        user_form.save()
        my_user_form.save()
        return HttpResponseRedirect(self.get_success_url())

    def form_invalid(self, user_form, my_user_form):
        return self.render_to_response(
            self.get_context_data(
                user_form=user_form,
                my_user_form=my_user_form))


class UserDelete(AddMyUserMixin, LoginRequiredMixin, NeverCacheFormMixin,
                 DeleteView):
    model = User
    template_name = 'users/delete.html'
    form_class = forms.UserForm
    my_user_form_class = forms.MyUserForm

    def get_success_url(self):
        return reverse('home')

    def get_object(self, **kwargs):
        obj = super().get_object(**kwargs)

        if self.request.user.is_authenticated():
            if obj != self.request.user:
                raise PermissionDenied

        return obj


class Home(AddMyUserMixin, ListView):
    template_name = 'network/home.html'
    model = models.Activity

    def get_queryset(self, **kwargs):

        if self.request.user.is_authenticated():
            my_user = models.MyUser.objects.get(user=self.request.user)

            query = Q()
            # Followed user performs an action
            query |= Q(user__following__followed=my_user)
            # Another user followed the requesting user
            query |= Q(followed_user=my_user)

            return self.model.objects.filter(query).order_by('-last_updated')

        else:
            return None


class ExplorePCA(TemplateView, AddMyUserMixin):
    template_name = 'explore/pca.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        user_dict = defaultdict(list)

        if context['login_user']:

            for trans in models.PCATransformedValues.objects.filter(
                    dataset__experiment__owners=context['login_user']):

                if trans.dataset.experiment.color:
                    colors = {'Default': trans.dataset.experiment.color}
                else:
                    colors = {
                        'Default': '#A9A9A9',
                        'Target': get_target_color(
                            trans.dataset.experiment.target),
                        'Cell type': get_cell_type_color(
                            trans.dataset.experiment.cell_type),
                    }

                user_dict[trans.pca.pk].append({
                    'colors': colors,
                    'dataset_name': trans.dataset.name,
                    'dataset_pk': trans.dataset.pk,
                    'experiment_cell_type': trans.dataset.experiment.cell_type,
                    'experiment_name': trans.dataset.experiment.name,
                    'experiment_pk': trans.dataset.experiment.pk,
                    'experiment_target': trans.dataset.experiment.target,
                    'transformed_values': trans.transformed_values,
                    'tags': {
                        'Cell type': trans.dataset.experiment.cell_type,
                        'Target': trans.dataset.experiment.target,
                    },
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


class ExploreNetwork(TemplateView, AddMyUserMixin):
    template_name = 'explore/network.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        network_lookup = dict()
        available_experiment_types = dict()
        available_organisms = dict()

        for org in models.Organism.objects.all():
            for exp_type in models.ExperimentType.objects.all():

                networks = models.OrganismNetwork.objects.filter(
                    organism=org, experiment_type=exp_type)
                if networks.exists():

                    try:
                        user_network = networks.get(
                            my_user=context['login_user'])
                        pk = user_network.pk
                    except models.OrganismNetwork.DoesNotExist:
                        default_network = networks.get(
                            my_user=None)
                        pk = default_network.pk

                    network_lookup['{}:{}'.format(
                        org.name,
                        exp_type.name,
                    )] = pk

                    if org.name not in available_experiment_types:
                        available_experiment_types[org.name] = []
                    if exp_type.name not in \
                            available_experiment_types[org.name]:
                        available_experiment_types[org.name].append(
                            exp_type.name)

                    if exp_type.name not in available_organisms:
                        available_organisms[exp_type.name] = []
                    if org.name not in available_organisms[exp_type.name]:
                        available_organisms[exp_type.name].append(org.name)

        context['network_lookup'] = network_lookup
        context['available_experiment_types'] = available_experiment_types
        context['available_organisms'] = available_organisms

        return context


class ExploreDendrogram(TemplateView, AddMyUserMixin):
    template_name = 'explore/dendrogram.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        dendrogram_lookup = dict()
        available_experiment_types = dict()
        available_organisms = dict()

        for org in models.Organism.objects.all():
            for exp_type in models.ExperimentType.objects.all():

                dendrograms = models.Dendrogram.objects.filter(
                    organism=org, experiment_type=exp_type)
                if dendrograms.exists():

                    try:
                        user_network = dendrograms.get(
                            my_user=context['login_user'])
                        pk = user_network.pk
                    except models.Dendrogram.DoesNotExist:
                        default_network = dendrograms.get(
                            my_user=None)
                        pk = default_network.pk

                    dendrogram_lookup['{}:{}'.format(
                        org.name,
                        exp_type.name,
                    )] = pk

                    if org.name not in available_experiment_types:
                        available_experiment_types[org.name] = []
                    if exp_type.name not in \
                            available_experiment_types[org.name]:
                        available_experiment_types[org.name].append(
                            exp_type.name)

                    if exp_type.name not in available_organisms:
                        available_organisms[exp_type.name] = []
                    if org.name not in available_organisms[exp_type.name]:
                        available_organisms[exp_type.name].append(org.name)

        context['dendrogram_lookup'] = dendrogram_lookup
        context['available_experiment_types'] = available_experiment_types
        context['available_organisms'] = available_organisms

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
        context['network'] = exp.get_network(my_user=context['login_user'])

        if context['login_user']:
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
        # context['network'] = dataset.get_network()

        if context['login_user']:
            context['is_favorite'] = models.Favorite.objects.filter(
                experiment=dataset.experiment,
                user=context['login_user'],
            ).exists()
            context['owned'] = dataset.experiment.owners.filter(
                pk=context['login_user'].pk).exists()

        return context


class MyUser(DetailView, AddMyUserMixin):
    template_name = 'users/user.html'
    model = User
    context_object_name = 'user_object'

    def get_object(self, **kwargs):
        user = super().get_object(**kwargs)
        obj = models.MyUser.objects.get(user=user)

        if not obj.is_public():
            if self.request.user.is_authenticated():
                if not obj.is_owned(self.request.user):
                    raise PermissionDenied
            else:
                raise PermissionDenied

        return user

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        my_user = models.MyUser.objects.get(user=self.get_object())
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

    def get_page_objs(self):
        paginator = Paginator(self.object_list,
                              self.get_paginate_by(self.object_list))
        page = self.request.GET.get('page')

        try:
            current_objects = paginator.page(page)
        except PageNotAnInteger:
            current_objects = paginator.page(1)
        except EmptyPage:
            current_objects = paginator.page(paginator.num_pages)

        return current_objects

    def get_queryset(self, base_query=None):

        query = Q(dataset__processed=True)  # Only returned processed exps
        if base_query:
            query &= base_query

        if self.form.is_valid():
            query &= self.form.get_query()

        return \
            models.Experiment.objects.filter(query).distinct().order_by('pk')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context['display_experiment_navbar'] = self.display_experiment_navbar
        context['experiment_counts'] = 0
        context['form'] = self.form
        context['search_field'] = self.form['search']
        context['other_fields'] = []
        for field in self.form:
            if field.name != 'search':
                context['other_fields'].append(field)

        context['page_objects'] = []

        for obj in self.get_page_objs():
            obj.plot_data = obj.get_average_metaplots()
            obj.meta_data = obj.get_metadata(self.my_user)
            obj.urls = obj.get_urls()
            context['page_objects'].append(obj)

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
            owners=self.my_user, dataset__processed=False, processed=False,
        ).distinct()
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
        self.target_user = User.objects.get(pk=self.kwargs['pk'])
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        my_user = models.MyUser.objects.get(user=self.target_user)
        base_query = Q(owners=my_user)
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

        rec_query = Q(user=self.my_user)
        if self.form.is_valid() and self.request.GET.get('rec_type'):
            rec_query &= Q(rec_type__in=self.form.get_rec_type())
        recommendations = models.Recommendation.objects.filter(rec_query)

        query = (
            ~Q(owners=self.my_user) & ~Q(favorite__user=self.my_user) &
            Q(recommended_experiment__in=recommendations)
        )

        if self.form.is_valid():
            query &= self.form.get_query()

        return self.model.objects.filter(query).distinct()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        for obj in context['page_objects']:
            obj.recommendation_tags = obj.get_recommendation_tags(self.my_user)
        return context


class SimilarExperiments(ExperimentList):
    model = models.Experiment
    template_name = 'experiments/similar_experiments.html'
    form_class = forms.SimilarExperimentFilterForm

    def get_queryset(self):

        exp = models.Experiment.objects.get(pk=self.kwargs['pk'])

        query = Q(public=True) & Q(sim_experiment_1__experiment_2=exp)

        if self.form.is_valid():
            query &= self.form.get_query()

        qs = self.model.objects.filter(query).distinct()
        self.sim_count = qs.count()

        return qs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context['ref_experiment'] = \
            models.Experiment.objects.get(pk=self.kwargs['pk'])
        context['sim_experiment_count'] = self.sim_count

        return context


class UserList(AddMyUserMixin, ListView):
    form_class = forms.UserFilterForm
    display_user_navbar = True

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

    def get_page_objs(self):
        paginator = Paginator(self.object_list,
                              self.get_paginate_by(self.object_list))
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

        return models.MyUser.objects.filter(query).distinct().order_by('pk')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context['display_user_navbar'] = self.display_user_navbar
        context['form'] = self.form
        context['search_field'] = self.form['search']
        context['other_fields'] = []
        for field in self.form:
            if field.name != 'search':
                context['other_fields'].append(field)

        context['page_objects'] = []

        for obj in self.get_page_objs():
            display_data = obj.get_display_data(self.my_user)

            obj.plot_data = display_data['plot_data']
            obj.meta_data = display_data['meta_data']
            obj.urls = display_data['urls']

            context['page_objects'].append(obj)

        return context


class AllUsers(UserList):
    template_name = 'users/all_users.html'

    def get_queryset(self):
        base_query = Q(public=True)
        if self.my_user:
            base_query &= ~Q(pk=self.my_user.pk)
        return super().get_queryset(base_query)


class Followed(LoginRequiredMixin, UserList):
    template_name = 'users/user_followed.html'

    def get_queryset(self):
        base_query = Q(followed__following=self.my_user)
        return super().get_queryset(base_query)


class Followers(LoginRequiredMixin, UserList):
    template_name = 'users/user_followers.html'

    def get_queryset(self):
        base_query = Q(following__followed=self.my_user)
        return super().get_queryset(base_query)


class UserFollowed(UserList):
    template_name = 'users/user_followed.html'
    display_user_navbar = False

    def get(self, request, *args, **kwargs):
        self.target_user = User.objects.get(pk=self.kwargs['pk'])
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        my_user = models.MyUser.objects.get(user=self.target_user)
        base_query = Q(followed__following=my_user)
        return super().get_queryset(base_query)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['target_user'] = self.target_user
        context['display_user_navbar'] = self.display_user_navbar
        return context


class UserFollowers(UserList):
    template_name = 'users/user_followers.html'
    display_user_navbar = False

    def get(self, request, *args, **kwargs):
        self.target_user = User.objects.get(pk=self.kwargs['pk'])
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        my_user = models.MyUser.objects.get(user=self.target_user)
        base_query = Q(following__followed=my_user)
        return super().get_queryset(base_query)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['target_user'] = self.target_user
        context['display_user_navbar'] = self.display_user_navbar
        return context
