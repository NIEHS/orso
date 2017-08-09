from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.views.generic \
    import View, TemplateView, DetailView, CreateView, UpdateView, DeleteView
from django.views.generic import ListView
from django.forms.models import inlineformset_factory
from django.db.models import Q, F  # noqa
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
    return Response(models.Dataset.get_browser_view(chromosome, start, end, datasets))


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


class Experiment(DetailView, AddMyUserMixin):
    template_name = 'network/experiment.html'
    model = models.Experiment

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        login_user = context['login_user']
        exp = self.get_object()

        context['selectable'] = dict()
        context['selectable']['personal'] = login_user.get_personal_experiment_ids()
        context['selectable']['favorite'] = login_user.get_favorite_experiment_ids()

        context['display_data'] = exp.get_display_data(context['login_user'])

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


class RecommendedExperiments(ExperimentList):
    model = models.Experiment
    template_name = 'network/recommended_experiments.html'
    form_class = forms.RecommendedExperimentFilterForm

    def get_queryset(self):

        query = Q(experimentrecommendation__owner=self.my_user)
        order_fields = [x[0] for x in self.form_class.order_choices]

        if self.form.is_valid():
            query &= self.form.get_query()
            order_fields = self.form.get_order()

        rank_eval = '+'.join(
            ['F(\'experimentrecommendation__{}\')'.format(x)
             for x in order_fields])
        if rank_eval:
            qs = (
                self.model.objects
                .filter(query)
                .annotate(rank=eval(rank_eval))
                .order_by('rank')
            )
        else:
            qs = self.model.objects.filter(query)

        for obj in qs:
            obj.plot_data = obj.get_average_metaplots()
            obj.meta_data = obj.get_metadata(self.my_user)
            obj.urls = obj.get_urls()

        return qs


class SimilarExperiments(ExperimentList):
    model = models.Experiment
    template_name = 'network/similar_experiments.html'
    form_class = forms.SimilarExperimentFilterForm
    correlation_model = None

    def get_queryset(self):

        exp = models.Experiment.objects.get(pk=self.kwargs['pk'])
        assemblies = \
            models.GenomeAssembly.objects.filter(dataset__experiment=exp)

        base_query = Q()
        for a in assemblies:
            base_query |= Q(dataset__assembly=a)

        form_query = Q()
        if self.form.is_valid():
            form_query &= self.form.get_query()
        qs = self.model.objects.filter(
            base_query & form_query).exclude(pk=exp.pk).distinct().all()

        for obj in qs:
            obj.plot_data = obj.get_average_metaplots()
            obj.meta_data = obj.get_metadata(self.my_user)
            obj.urls = obj.get_urls()
            obj.score = self.correlation_model.get_score(exp, obj)

        return sorted(qs, key=lambda x: x.score)


class SimilarValuesExperiments(SimilarExperiments):
    correlation_model = models.ExperimentCorrelation


class SimilarMetadataExperiments(SimilarExperiments):
    correlation_model = models.MetadataCorrelation


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


def dataset_comparison(request, x, y):
    dataset_x = get_object_or_404(models.Dataset, pk=x)
    dataset_y = get_object_or_404(models.Dataset, pk=y)
    return render(
        request, 'network/dataset_comparison.html',
        {'dataset_x': dataset_x, 'dataset_y': dataset_y})


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
