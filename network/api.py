from rest_framework import viewsets
from rest_framework.exceptions import NotAcceptable
from rest_framework.response import Response
from rest_framework.decorators import detail_route
from django.contrib.auth.models import User
from django.http import HttpResponse

from . import models
from . import serializers


def try_int(val, default=None):
    """Return int or default value."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


class ExperimentViewset(viewsets.ModelViewSet):

    def get_queryset(self):
        return models.Experiment.objects.all()

    def get_serializer_class(self):
        return serializers.ExperimentSerializer

    @detail_route(methods=['get'], url_path='add-favorite')
    def add_experiment_to_favorites(self, request, pk=None):
        object = self.get_object()
        my_user = models.MyUser.objects.get(user=self.request.user)

        models.Favorite.objects.update_or_create(
            user=my_user,
            experiment=object,
        )

        return HttpResponse(status=202)

    @detail_route(methods=['get'], url_path='remove-favorite')
    def remove_experiment_from_favorites(self, request, pk=None):
        object = self.get_object()
        my_user = models.MyUser.objects.get(user=self.request.user)

        models.Favorite.objects.get(
            user=my_user,
            experiment=object,
        ).delete()

        return HttpResponse(status=202)

    @detail_route(methods=['get'], url_path='hide-recommendation')
    def hide_experiment_recommendation(self, request, pk=None):
        object = self.get_object()
        my_user = models.MyUser.objects.get(user=self.request.user)
        recommendation = models.ExperimentRecommendation.objects.get(
            owner=my_user, recommended=object)
        recommendation.hidden = True
        recommendation.save()
        return HttpResponse(status=202)


class DatasetViewset(viewsets.ModelViewSet):

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

        favorite = models.DataFavorite.objects.get(
            owner=my_user, favorite=object)
        favorite.delete()

        return HttpResponse(status=202)

    @detail_route(methods=['get'], url_path='hide-recommendation')
    def hide_dataset_recommendation(self, request, pk=None):
        object = self.get_object()
        my_user = models.MyUser.objects.get(user=self.request.user)
        recommendation = models.DataRecommendation.objects.get(
            owner=my_user, recommended=object)
        recommendation.hidden = True
        recommendation.save()
        return HttpResponse(status=202)


class UserViewset(viewsets.ModelViewSet):

    def get_queryset(self):
        return User.objects.all()

    @detail_route(methods=['get'], url_path='follow')
    def add_user_to_favorites(self, request, pk=None):

        followed_user = models.MyUser.objects.get(user=self.get_object())
        following_user = models.MyUser.objects.get(user=self.request.user)

        models.Follow.objects.update_or_create(
            followed=followed_user,
            following=following_user,
        )

        return HttpResponse(status=202)

    @detail_route(methods=['get'], url_path='stop-following')
    def remove_user_from_favorites(self, request, pk=None):

        followed_user = models.MyUser.objects.get(user=self.get_object())
        following_user = models.MyUser.objects.get(user=self.request.user)

        try:
            follow = models.Follow.objects.get(
                followed=followed_user,
                following=following_user,
            )
            follow.delete()
        except models.Follow.DoesNotExist:
            pass

        return HttpResponse(status=202)

    @detail_route(methods=['get'], url_path='hide-recommendation')
    def hide_user_recommendation(self, request, pk=None):
        object = self.get_object()
        my_user = models.MyUser.objects.get(user=self.request.user)
        recommendation = models.UserRecommendation.objects.get(
            owner=my_user, recommended=object)
        recommendation.hidden = True
        recommendation.save()
        return HttpResponse(status=202)


class PCAPlotViewset(viewsets.ModelViewSet):
    queryset = models.PCA.objects.all()
    serializer_class = serializers.PCAPlotSerializer


class MetaPlotViewset(viewsets.ModelViewSet):
    queryset = models.MetaPlot.objects.all()
    serializer_class = serializers.MetaPlotSerializer


class FeatureValuesViewset(viewsets.ModelViewSet):
    queryset = models.FeatureValues.objects.all()
    serializer_class = serializers.FeatureValuesSerializer


class NetworkViewset(viewsets.ModelViewSet):
    queryset = models.OrganismNetwork.objects.all()
    serializer_class = serializers.NetworkSerializer


class DendrogramViewset(viewsets.ModelViewSet):
    queryset = models.Dendrogram.objects.all()
    serializer_class = serializers.DendrogramSerializer
