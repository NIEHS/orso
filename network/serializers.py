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


class PCAPlotSerializer(serializers.ModelSerializer):
    pca_plot = serializers.SerializerMethodField('_pca_plot')
    explained_variance = \
        serializers.SerializerMethodField('_explained_variance')
    components = serializers.SerializerMethodField('_components')

    def _pca_plot(self, pca):

        plot = []

        datasets = models.Dataset.objects.filter(pcatransformedvalues__pca=pca)
        for ds in datasets:
            plot.append({
                'experiment_name': ds.experiment.name,
                'dataset_name': ds.name,
                'experiment_pk': ds.experiment.pk,
                'dataset_pk': ds.pk,
                'experiment_cell_type': ds.experiment.cell_type,
                'experiment_target': ds.experiment.target,
                'transformed_values':
                    ds.pcatransformedvalues_set
                      .get(pca=pca).transformed_values,
            })

        return plot

    def _explained_variance(self, pca):
        return pca.pca.explained_variance_ratio_

    def _components(self, pca):
        components = []

        if pca.locus_group.group_type in ['genebody', 'promoter', 'mRNA']:
            locus_names = [
                x.locus.transcript.gene.name for x in
                models.PCALocusOrder.objects.filter(pca=pca).order_by('order')
            ]
        elif pca.locus_group.group_type in ['enhancer']:
            locus_names = [
                x.locus.enhancer.name for x in
                models.PCALocusOrder.objects.filter(pca=pca).order_by('order')
            ]

        for _component in pca.pca.components_:
            components.append(
                sorted(
                    zip(locus_names, _component),
                    key=lambda x: -abs(x[1]),
                )[:20]
            )

        return components

    class Meta:
        model = models.PCA
        fields = ('pca_plot', 'explained_variance', 'components')
