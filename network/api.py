from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import detail_route
from django.http import HttpResponse

from . import models
from . import serializers


class DatasetViewset(viewsets.ModelViewSet):

    @detail_route(methods=['get'])
    def browser_view(self, request, pk=None):
        query = self.request.GET.get('query')
        object = self.get_object()
        return Response(object.get_browser_view(query))

    @detail_route(methods=['get'], url_path='promoter-intersection')
    def promoter_intersection(self, request, pk=None):
        object = self.get_object()
        return Response(object.promoter_intersection.intersection_values)

    def get_queryset(self):
        return models.Dataset.objects.all()

    def get_serializer_class(self):
        return serializers.DatasetSerializer

    @detail_route(methods=['get'], url_path='add-favorite')
    def add_dataset_to_favorites(self, request, pk=None):
        object = self.get_object()
        my_user = models.MyUser.objects.get(user=self.request.user)

        models.DataFavorite.objects.create(
            owner=my_user,
            favorite=object,
        )

        return HttpResponse(status=202)

    @detail_route(methods=['get'], url_path='remove-favorite')
    def remove_dataset_from_favorites(self, request, pk=None):
        object = self.get_object()
        my_user = models.MyUser.objects.get(user=self.request.user)

        favorite = models.DataFavorite.objects.get(owner=my_user, favorite=object)
        favorite.delete()

        return HttpResponse(status=202)

    @detail_route(methods=['get'], url_path='hide-recommendation')
    def hide_dataset_recommendation(self, request, pk=None):
        object = self.get_object()
        my_user = models.MyUser.objects.get(user=self.request.user)
        recommendation = models.DataRecommendation.objects.get(owner=my_user, recommended=object)
        recommendation.hidden = True
        recommendation.save()
        return HttpResponse(status=202)


class UserViewset(viewsets.ModelViewSet):

    def get_queryset(self):
        return models.MyUser.objects.all()

    @detail_route(methods=['get'], url_path='add-favorite')
    def add_user_to_favorites(self, request, pk=None):
        object = self.get_object()
        my_user = models.MyUser.objects.get(user=self.request.user)

        models.UserFavorite.objects.create(
            owner=my_user,
            favorite=object,
        )

        return HttpResponse(status=202)

    @detail_route(methods=['get'], url_path='remove-favorite')
    def remove_user_from_favorites(self, request, pk=None):
        object = self.get_object()
        my_user = models.MyUser.objects.get(user=self.request.user)

        favorite = models.UserFavorite.objects.get(owner=my_user, favorite=object)
        favorite.delete()

        return HttpResponse(status=202)

    @detail_route(methods=['get'], url_path='hide-recommendation')
    def hide_user_recommendation(self, request, pk=None):
        object = self.get_object()
        my_user = models.MyUser.objects.get(user=self.request.user)
        recommendation = models.UserRecommendation.objects.get(owner=my_user, recommended=object)
        recommendation.hidden = True
        recommendation.save()
        return HttpResponse(status=202)
