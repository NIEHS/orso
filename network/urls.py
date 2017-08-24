from django.conf.urls import include, url
from rest_framework.routers import DefaultRouter

from . import api, views

router = DefaultRouter()
router.register('experiment',
                api.ExperimentViewset,
                base_name='experiment')
router.register('user',
                api.UserViewset,
                base_name='user')
router.register('browser',
                api.BrowserViewset,
                base_name='browser')
router.register('pca-plot',
                api.PCAPlotViewset,
                base_name='pca-plot')

urlpatterns = [

    url(r'^$', views.Index.as_view(), name='index'),
    url(r'^home/$', views.Home.as_view(), name='home'),
    url(r'^experiment/(?P<pk>\d+)/$', views.Experiment.as_view(),
        name='experiment'),
    url(r'^gene/(?P<pk>\d+)/$', views.Gene.as_view(),
        name='gene'),
    url(r'^transcript/(?P<pk>\d+)/$', views.Transcript.as_view(),
        name='transcript'),

    url(r'^explore/$', views.Explore.as_view(), name='explore'),

    url(r'^experiment/(?P<pk>\d+)/similar_values/$',
        views.SimilarValuesExperiments.as_view(),
        name='similar_values'),
    url(r'^experiment/(?P<pk>\d+)/similar_metadata/$',
        views.SimilarMetadataExperiments.as_view(),
        name='similar_metadata'),

    url(r'^user/(?P<pk>\d+)/$', views.MyUser.as_view(), name='user'),

    url(r'^personal_experiments/$', views.PersonalExperiments.as_view(),
        name='personal_experiments'),
    url(r'^favorite_experiments/$', views.FavoriteExperiments.as_view(),
        name='favorite_experiments'),
    url(r'^recommended_experiments/$', views.RecommendedExperiments.as_view(),
        name='recommended_experiments'),

    url(r'^favorite_users/$', views.FavoriteUsers.as_view(),
        name='favorite_users'),
    url(r'^recommended_users/$', views.RecommendedUsers.as_view(),
        name='recommended_users'),
    url(r'^dataset-comparison/(?P<x>\d+)-(?P<y>\d+)/$',
        views.DatasetComparison.as_view(), name='dataset-comparison'),
    url(r'^experiment-comparison/(?P<x>\d+)-(?P<y>\d+)/$',
        views.ExperimentComparison.as_view(), name='experiment-comparison'),

    url(r'^create_experiment/$', views.ExperimentCreate.as_view(),
        name='create_experiment'),
    url(r'^update_experiment/(?P<pk>\d+)/$', views.ExperimentUpdate.as_view(),
        name='update_experiment'),
    url(r'^delete_experiment/(?P<pk>\d+)/$', views.ExperimentDelete.as_view(),
        name='delete_experiment'),

    #  For PCA, PK refers to the GenomicRegions object
    url(r'^pca/(?P<pk>\d+)/$', views.PCA.as_view(), name='pca'),

    # url(r'^test/$', views.TestSmallDataView.as_view(), name='test'),
    url(r'^test/$', views.TestSmallUserView.as_view(), name='test'),
    url(r'^browser/$', views.browser, name='browser'),

    url(r'^api/',
        include(router.urls, namespace='api')),

    url(r'^selectable/', include('selectable.urls')),
]
