from django.conf.urls import include, url
from rest_framework.routers import DefaultRouter

from . import api, views

router = DefaultRouter()
router.register('dataset',
                api.DatasetViewset,
                base_name='dataset')
router.register('user',
                api.UserViewset,
                base_name='user')

urlpatterns = [

    url(r'^$', views.Index.as_view(), name='index'),
    url(r'^home/$', views.Home.as_view(), name='home'),
    url(r'^dataset/(?P<pk>\d+)/$', views.Dataset.as_view(), name='dataset'),
    url(r'^user/(?P<pk>\d+)/$', views.MyUser.as_view(), name='user'),

    url(r'^personal_datasets/$', views.PersonalDatasets.as_view(), name='personal_datasets'),
    url(r'^favorite_datasets/$', views.FavoriteDatasets.as_view(), name='favorite_datasets'),
    url(r'^recommended_datasets/$', views.RecommendedDatasets.as_view(),
        name='recommended_datasets'),

    url(r'^favorite_users/$', views.FavoriteUsers.as_view(), name='favorite_users'),
    url(r'^recommended_users/$', views.RecommendedUsers.as_view(), name='recommended_users'),
    url(r'^dataset_comparison/(?P<x>\d+)-(?P<y>\d+)/$',
        views.dataset_comparison, name='dataset_comparison'),

    url(r'^create_dataset/$', views.DatasetCreate.as_view(), name='create_dataset'),
    url(r'^update_dataset/(?P<pk>\d+)/$', views.DatasetUpdate.as_view(), name='update_dataset'),
    url(r'^delete_dataset/(?P<pk>\d+)/$', views.DatasetDelete.as_view(), name='delete_dataset'),

    # url(r'^test/$', views.TestSmallDataView.as_view(), name='test'),
    url(r'^test/$', views.TestSmallUserView.as_view(), name='test'),

    url(r'^api/',
        include(router.urls, namespace='api')),
]
