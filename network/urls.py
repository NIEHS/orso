from django.conf.urls import url
from . import views


urlpatterns = [
    url(r'^$', views.Home.as_view(), name='index'),
    url(r'^dataset/(?P<pk>\d+)/$', views.dataset, name='dataset'),
]
