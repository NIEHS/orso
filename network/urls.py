from django.conf.urls import include, url
from django.views.generic.base import RedirectView
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
router.register('network-plot',
                api.NetworkViewset,
                base_name='network-plot')
router.register('metaplot',
                api.MetaPlotViewset,
                base_name='metaplot')
router.register('feature-values',
                api.FeatureValuesViewset,
                base_name='feature-values')

urlpatterns = [

    url(r'^home/$', views.Home.as_view(), name='home'),
    url(r'^dataset/(?P<pk>\d+)/$', views.Dataset.as_view(),
        name='dataset'),
    url(r'^experiment/(?P<pk>\d+)/$', views.Experiment.as_view(),
        name='experiment'),
    url(r'^gene/(?P<pk>\d+)/$', views.Gene.as_view(),
        name='gene'),
    url(r'^transcript/(?P<pk>\d+)/$', views.Transcript.as_view(),
        name='transcript'),

    url(r'^explore/pca/$', views.ExplorePCA.as_view(), name='explore_pca'),
    # url(r'^explore/network/$', views.ExploreNetwork.as_view(),
    #     name='explore_network'),
    url(r'^explore/overview/$', views.ExploreOverview.as_view(),
        name='explore_overview'),
    url(r'^explore/recommendations/$', views.ExploreRecommendations.as_view(),
        name='explore_recommendations'),

    url(r'^experiment/(?P<pk>\d+)/similar_experiments/$',
        views.SimilarExperiments.as_view(), name='similar_experiments'),

    url(r'^user/(?P<pk>\d+)/$', views.MyUser.as_view(), name='user'),

    url(r'^experiments/$', views.AllExperiments.as_view(),
        name='all_experiments'),
    url(r'^personal_experiments/$', views.PersonalExperiments.as_view(),
        name='personal_experiments'),
    url(r'^favorite_experiments/$', views.FavoriteExperiments.as_view(),
        name='favorite_experiments'),
    url(r'^recommended_experiments/$', views.RecommendedExperiments.as_view(),
        name='recommended_experiments'),
    url(r'^user_experiments/(?P<pk>\d+)/$', views.UserExperiments.as_view(),
        name='user_experiments'),

    url(r'^users/$', views.AllUsers.as_view(),
        name='all_users'),
    url(r'^followed/$', views.Followed.as_view(),
        name='followed'),
    url(r'^followers/$', views.Followers.as_view(),
        name='followers'),
    url(r'^user_followed/(?P<pk>\d+)/$', views.UserFollowed.as_view(),
        name='user_followed'),
    url(r'^user_followers/(?P<pk>\d+)/$', views.UserFollowers.as_view(),
        name='user_followers'),

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

    url(r'^update_user/(?P<pk>\d+)/$', views.UserUpdate.as_view(),
        name='update_user'),
    url(r'^delete_user/(?P<pk>\d+)/$', views.UserDelete.as_view(),
        name='delete_user'),

    url(r'^pca/(?P<pk>\d+)/$', views.PCA.as_view(), name='pca'),

    # url(r'^test/$', views.TestSmallDataView.as_view(), name='test'),
    url(r'^test/$', views.TestSmallUserView.as_view(), name='test'),
    url(r'^browser/$', views.browser, name='browser'),
    # url(r'^network/$', views.network, name='network'),

    url(r'^api/',
        include(router.urls, namespace='api')),

    url(r'^selectable/', include('selectable.urls')),
    url(r'^$', RedirectView.as_view(pattern_name='home', permanent=False)),
]
