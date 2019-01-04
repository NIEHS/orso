import json
import os
import subprocess
from tempfile import NamedTemporaryFile

from django.conf import settings
import pandas as pd

from network import models

BIGWIG_AVERAGE_OVER_BED_PATH = \
    os.path.join(settings.BIN_DIR, 'bigWigAverageOverBed')


def call_bigwig_average_over_bed(bigwig_name, bed_name, out_name):
    '''
    Call Kent tools bigWigAverageOverBed.
    '''
    FNULL = open(os.devnull, 'w')
    cmd = [
        BIGWIG_AVERAGE_OVER_BED_PATH,
        bigwig_name,
        bed_name,
        out_name,
    ]
    print('Running subprocess: {}'.format(' '.join(cmd)))
    subprocess.call(cmd, stdout=FNULL, stderr=subprocess.STDOUT)


def generate_intersection_df(locus_group, experiment_type, datasets=None,
                             loci=None):
    '''
    For a given LocusGroup, generate a pandas DF with intersection values.
    '''
    if datasets is None:
        datasets = models.Dataset.objects.all()

    d = {}
    for intersection in models.DatasetIntersectionJson.objects.filter(
        locus_group=locus_group,
        dataset__experiment__experiment_type=experiment_type,
        dataset__in=datasets,
    ):
        values = json.loads(intersection.intersection_values)
        series = pd.Series(
            values['normalized_values'], index=values['locus_pks'])
        d.update({intersection.dataset.pk: series})
    df = pd.DataFrame(d)

    if loci:
        df = df.loc[[x.pk for x in loci]]

    return df


def generate_pca_transformed_df(pca, datasets=None):
    if not datasets:
        datasets = models.Dataset.objects.all()

    d = {}
    for transformed_values in models.PCATransformedValues.objects.filter(
        pca=pca,
        dataset__in=datasets,
    ):
        series = pd.Series(transformed_values.transformed_values)
        d.update({transformed_values.dataset.pk: series})
    df = pd.DataFrame(d)
    df.sort_index(axis=1, inplace=True)

    return df


def download_dataset_bigwigs(datasets, check_certificate=True):

    os.makedirs(settings.BIGWIG_TEMP_DIR, exist_ok=True)

    download_list_file = NamedTemporaryFile(mode='w')
    bigwig_paths = dict()

    for dataset in datasets:

        paths = dataset.generate_local_bigwig_paths()

        if dataset.is_stranded():

            download_list_file.write('{}\n'.format(dataset.plus_url))
            download_list_file.write(
                '\tdir={}\n'.format(os.path.dirname(paths['plus'])))
            download_list_file.write(
                '\tout={}\n'.format(os.path.basename(paths['plus'])))

            download_list_file.write('{}\n'.format(dataset.minus_url))
            download_list_file.write(
                '\tdir={}\n'.format(os.path.dirname(paths['minus'])))
            download_list_file.write(
                '\tout={}\n'.format(os.path.basename(paths['minus'])))

        else:

            download_list_file.write('{}\n'.format(dataset.ambiguous_url))
            download_list_file.write(
                '\tdir={}\n'.format(os.path.dirname(paths['ambiguous'])))
            download_list_file.write(
                '\tout={}\n'.format(os.path.basename(paths['ambiguous'])))

        bigwig_paths[dataset.pk] = paths

    download_list_file.flush()

    download_complete = False
    while not download_complete:
        return_code = subprocess.call([
            'aria2c',
            '--allow-overwrite=true',
            '--conditional-get=true',
            '--check-certificate={}'.format(str(check_certificate).lower()),
            '-x', '16',
            '-s', '16',
            '-i', download_list_file.name,
        ])
        print('aria2c exitted with return code {}.'.format(
            str(return_code)))
        if return_code == 0:
            download_complete = True

    download_list_file.close()

    return bigwig_paths


def remove_dataset_bigwigs(datasets):

    for dataset in datasets:
        paths = dataset.generate_local_bigwig_paths()
        for path in paths.values():
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
