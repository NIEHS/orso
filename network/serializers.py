from rest_framework import serializers

from . import models


class ExperimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Dataset
        # exclude = (
        #     'promoter_intersection', 'promoter_metaplot',
        #     'enhancer_intersection', 'enhancer_metaplot',)


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Dataset
        exclude = (
            'promoter_intersection', 'promoter_metaplot',
            'enhancer_intersection', 'enhancer_metaplot',)
