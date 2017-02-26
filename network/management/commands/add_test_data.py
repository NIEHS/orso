from django.core.management.base import BaseCommand
from network import models
import csv
import json
import os


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('metadata_csv', type=str)

    def handle(self, *args, **options):
        curr_dir = os.getcwd()
        assembly = models.GenomeAssembly.objects.get(name='mm9')
        with open(options['metadata_csv']) as metadata:
            reader = csv.DictReader(metadata, delimiter=',')
            for r in reader:
                url = 'http://snpinfo.niehs.nih.gov/ucscview/andy/personal_files/' + r['bigWig']  # noqa
                name = r['name']
                data_type = 'ChIP-seq'
                cell_type = r['cell_type']
                target = r['antibody']

                exp = models.Experiment.objects.create(
                    name=name,
                    slug=name,
                    data_type=data_type,
                    cell_type=cell_type,
                    target=target,
                )
                ds = models.Dataset.objects.create(
                    experiment=exp,
                    ambiguous_url=url,
                    name=name,
                    slug=name,
                    assembly=assembly,
                )

                promoter_bed = curr_dir + '/data/genomic_regions/mm9_RefSeq_promoters.bed'  # noqa
                enhancer_bed = curr_dir + '/data/genomic_regions/mm9_vista_enhancers.bed'  # noqa

                promoter_gr = models.GenomicRegions.objects.filter(name__startswith='mm9_RefSeq_promoters')[0]  # noqa
                enhancer_gr = models.GenomicRegions.objects.filter(name__startswith='mm9_vista_enhancers')[0]  # noqa

                promoter_intersection = curr_dir + '/data/intersections/' + r['header'] + '.promoters.intersection.json'  # noqa
                enhancer_intersection = curr_dir + '/data/intersections/' + r['header'] + '.enhancers.intersection.json'  # noqa

                promoter_metaplot = curr_dir + '/data/metaplots/' + r['header'] + '.promoters.metaplot.json'  # noqa
                enhancer_metaplot = curr_dir + '/data/metaplots/' + r['header'] + '.enhancers.metaplot.json'  # noqa

                with open(promoter_metaplot) as f:
                    models.MetaPlot.objects.create(
                        genomic_regions=promoter_gr,
                        dataset=ds,
                        meta_plot=json.load(f),
                    )
                with open(enhancer_metaplot) as f:
                    models.MetaPlot.objects.create(
                        genomic_regions=enhancer_gr,
                        dataset=ds,
                        meta_plot=json.load(f),
                    )
                with open(promoter_intersection) as f:
                    models.IntersectionValues.objects.create(
                        genomic_regions=promoter_gr,
                        dataset=ds,
                        intersection_values=json.load(f),
                    )
                with open(enhancer_intersection) as f:
                    models.IntersectionValues.objects.create(
                        genomic_regions=enhancer_gr,
                        dataset=ds,
                        intersection_values=json.load(f),
                    )
