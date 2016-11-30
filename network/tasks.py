from celery.decorators import task
# from django.apps import apps

from analysis.metaplot import MetaPlot
from . import models


@task()
def process_dataset(id_):
    print('Processing dataset {}...'.format(str(id_)))
    # dataset = apps.get_model('network', 'Dataset').objects.get(id=id_)
    dataset = models.Dataset.objects.get(id=id_)

    promoter_regions = dataset.assembly.default_annotation.promoters
    enhancer_regions = dataset.assembly.default_annotation.enhancers

    pm = MetaPlot(
        promoter_regions.bed_file.path,
        single_bw=dataset.ambiguous_url
    )

    em = MetaPlot(
        enhancer_regions.bed_file.path,
        single_bw=dataset.ambiguous_url,
    )

    dataset.promoter_metaplot = models.MetaPlot.objects.create(
        genomic_regions=promoter_regions,
        bigwig_url=dataset.ambiguous_url,
        relative_start=-2500,
        relative_end=2499,
        meta_plot=pm.create_metaplot_json(),
    )
    dataset.enhancer_metaplot = models.MetaPlot.objects.create(
        genomic_regions=enhancer_regions,
        bigwig_url=dataset.ambiguous_url,
        relative_start=-2500,
        relative_end=2499,
        meta_plot=em.create_metaplot_json(),
    )
    dataset.promoter_intersection = models.IntersectionValues.objects.create(
        genomic_regions=promoter_regions,
        bigwig_url=dataset.ambiguous_url,
        relative_start=-2500,
        relative_end=2499,
        intersection_values=pm.create_intersection_json(),
    )
    dataset.enhancer_intersection = models.IntersectionValues.objects.create(
        genomic_regions=enhancer_regions,
        bigwig_url=dataset.ambiguous_url,
        relative_start=-2500,
        relative_end=2499,
        intersection_values=em.create_intersection_json(),
    )

    dataset.save()
