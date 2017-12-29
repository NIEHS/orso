import json
import os
from subprocess import call
from tempfile import NamedTemporaryFile

from celery import group
from celery.decorators import task

from analysis import metaplot, transcript_coverage
from analysis.normalization import normalize_locus_intersection_values
from network import models


@task()
def process_datasets(datasets, chunk=100):

    assembly_to_intersection_bed = dict()
    for ds in datasets:
        if ds.assembly not in assembly_to_intersection_bed:
            assembly_to_intersection_bed[ds.assembly] = dict()
            for lg in models.LocusGroup.objects.filter(assembly=ds.assembly):
                bed = NamedTemporaryFile(mode='w', delete=False)
                transcript_coverage.generate_locusgroup_bed(lg, bed)
                assembly_to_intersection_bed[ds.assembly][lg] = bed

    assembly_to_metaplot_bed = dict()
    for ds in datasets:
        if ds.assembly not in assembly_to_metaplot_bed:
            assembly_to_metaplot_bed[ds.assembly] = dict()
            for lg in models.LocusGroup.objects.filter(assembly=ds.assembly):
                bed = NamedTemporaryFile(mode='w', delete=False)
                metaplot.generate_metaplot_bed(lg, bed)
                assembly_to_metaplot_bed[ds.assembly][lg] = bed

    bigwig_dir = os.path.join(os.getcwd(), 'data', 'bigwig_temp')
    os.makedirs(bigwig_dir, exist_ok=True)

    for i in range(0, len(datasets), chunk):
        index_1 = i
        index_2 = min(i + chunk, len(datasets))
        dataset_chunk = datasets[index_1:index_2]

        download_list_file = NamedTemporaryFile(mode='w')
        bigwig_paths = dict()
        for ds in dataset_chunk:
            if ds.is_stranded():

                download_list_file.write('{}\n'.format(ds.plus_url))
                download_list_file.write('\tdir={}\n'.format(bigwig_dir))
                download_list_file.write('{}\n'.format(ds.minus_url))
                download_list_file.write('\tdir={}\n'.format(bigwig_dir))

                plus_local_path = os.path.join(
                    bigwig_dir, os.path.basename(ds.plus_url))
                minus_local_path = os.path.join(
                    bigwig_dir, os.path.basename(ds.minus_url))

                bigwig_paths[ds.pk] = {
                    'ambiguous': None,
                    'plus': plus_local_path,
                    'minus': minus_local_path,
                }

            else:

                download_list_file.write('{}\n'.format(ds.ambiguous_url))
                download_list_file.write('\tdir={}\n'.format(bigwig_dir))

                ambiguous_local_path = os.path.join(
                    bigwig_dir, os.path.basename(ds.ambiguous_url))

                bigwig_paths[ds.pk] = {
                    'ambiguous': ambiguous_local_path,
                    'plus': None,
                    'minus': None,
                }

        download_list_file.flush()
        call([
            'aria2c',
            '--allow-overwrite=true',
            '--conditional-get=true',
            '-x', '16',
            '-s', '16',
            '-i', download_list_file.name,
        ])
        download_list_file.close()

        tasks = []
        for ds in dataset_chunk:
            intersection_beds = assembly_to_intersection_bed[ds.assembly]
            for lg, bed in intersection_beds.items():
                tasks.append(process_dataset_intersection.s(
                    ds.pk,
                    lg.pk,
                    bed.name,
                    bigwig_paths[ds.pk],
                ))

            metaplot_beds = assembly_to_metaplot_bed[ds.assembly]
            for lg, bed in metaplot_beds.items():
                tasks.append(process_dataset_metaplot.s(
                    ds.pk,
                    lg.pk,
                    bed.name,
                    bigwig_paths[ds.pk],
                ))

        job = group(tasks)
        results = job.apply_async()
        results.join()

        for ds in dataset_chunk:
            ds.processed = True
            ds.save()

        for paths in bigwig_paths.values():
            for field in ['ambiguous', 'plus', 'minus']:
                if paths[field]:
                    os.remove(paths[field])

    for bed_dict in assembly_to_intersection_bed.values():
        for bed in bed_dict.values():
            bed.close()
    for bed_dict in assembly_to_metaplot_bed.values():
        for bed in bed_dict.values():
            bed.close()


@task()
def process_dataset_intersection(dataset_pk, locusgroup_pk, bed_path, bigwigs):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    locus_group = models.LocusGroup.objects.get(pk=locusgroup_pk)

    loci = models.Locus.objects.filter(group=locus_group).order_by('pk')
    locus_values = transcript_coverage.get_locus_values(
        loci,
        bed_path,
        ambiguous_bigwig=bigwigs['ambiguous'],
        plus_bigwig=bigwigs['plus'],
        minus_bigwig=bigwigs['minus'],
    )
    normalized_values = \
        normalize_locus_intersection_values(loci, locus_values)
    intersection_values = {
        'locus_pks': [],
        'raw_values': [],
        'normalized_values': [],
    }
    for locus in loci:
        intersection_values['locus_pks'].append(locus.pk)
        intersection_values['raw_values'].append(locus_values[locus])
        intersection_values['normalized_values'].append(
            normalized_values[locus])
    models.DatasetIntersectionJson.objects.update_or_create(
        dataset=dataset,
        locus_group=locus_group,
        defaults={
            'intersection_values': json.dumps(intersection_values),
        }
    )


@task()
def process_dataset_metaplot(dataset_pk, locusgroup_pk, bed_path, bigwigs):
    dataset = models.Dataset.objects.get(pk=dataset_pk)
    locus_group = models.LocusGroup.objects.get(pk=locusgroup_pk)
    metaplot_out = metaplot.get_metaplot_values(
        locus_group,
        bed_path=bed_path,
        ambiguous_bigwig=bigwigs['ambiguous'],
        plus_bigwig=bigwigs['plus'],
        minus_bigwig=bigwigs['minus'],
    )
    models.MetaPlot.objects.update_or_create(
        dataset=dataset,
        locus_group=locus_group,
        defaults={
            'metaplot': json.dumps(metaplot_out),
        },
    )
