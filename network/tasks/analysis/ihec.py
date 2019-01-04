import json
import re
from pprint import pprint
from subprocess import call, DEVNULL

from celery import group

from network import models
from network.tasks.analysis.utils import download_dataset_bigwigs
from network.tasks.processing import process_dataset_batch

chunk = 100

histone_marks = [
    'H3K27ac',
    'H3K27me3',
    'H3K36me3',
    'H3K4me1',
    'H3K4me3',
    'H3K9me3',
    'Input',
]

skip_assays = ['WGB-Seq']
skip_projects = ['ENCODE', 'NIH Roadmap']


def check_download(download_url):
    return call(['aria2c', '--dry-run', '--check-certificate=false',
                 download_url]) == 0


def add_ihec(json_path):

    with open(json_path) as f:
        ihec_dict = json.load(f)

    # Read assembly
    assembly_name = ihec_dict['hub_description']['assembly']
    if assembly_name == 'hg38':
        assembly_name = 'GRCh38'
    assembly_obj = models.Assembly.objects.get(name=assembly_name)

    organism_obj = models.Organism.objects.get(assembly=assembly_obj)

    experiments = []
    datasets = []

    # Iterate through 'datasets'
    for dataset_name, info in list(ihec_dict['datasets'].items()):

        skip = False

        project_name = info['ihec_data_portal']['publishing_group']
        cell_type = info['ihec_data_portal']['cell_type']
        assay_type = info['ihec_data_portal']['assay']

        sample_id = info['sample_id']

        # projects.add(project_name)
        forward_bigwig = None
        reverse_bigwig = None
        unstranded_bigwig = None
        if all([
            'signal_forward' in info['browser'],
            'signal_reverse' in info['browser'],
        ]):
            forward_bigwig = \
                info['browser']['signal_forward'][0]['big_data_url']
            reverse_bigwig = \
                info['browser']['signal_reverse'][0]['big_data_url']
        elif 'signal_unstranded' in info['browser']:
            unstranded_bigwig = \
                info['browser']['signal_unstranded'][0]['big_data_url']
        elif 'signal' in info['browser']:
            unstranded_bigwig = \
                info['browser']['signal'][0]['big_data_url']

        if not any([forward_bigwig and reverse_bigwig, unstranded_bigwig]):
            skip = True
        else:
            if forward_bigwig and reverse_bigwig:
                if not all([
                    check_download(forward_bigwig),
                    check_download(reverse_bigwig),
                ]):
                    skip = True
            else:
                if not check_download(unstranded_bigwig):
                    skip = True

        protein_target = None
        if assay_type in histone_marks:
            protein_target = assay_type
            assay_type = 'ChIP-seq'
        elif assay_type in ['mRNA-Seq', 'RNA-Seq']:
            assay_type = 'RNA-seq'

        if assay_type in skip_assays:
            skip = True
        if project_name in skip_projects:
            skip = True

        name_components = []
        name_components.append(sample_id)
        name_components.append(assay_type.replace('-', ''))
        if protein_target:
            name_components.append(protein_target)
        name_components.append(''.join(
            [x.capitalize() for x in re.split('\W+|_', cell_type)]))

        sample_name = '-'.join(name_components)

        description = None

        if not skip:

            project_obj = models.Project.objects.get_or_create(
                name=project_name,
            )[0]

            experiment_type_obj = models.ExperimentType.objects.get(
                name=assay_type,
            )

            exp_obj, exp_created = models.Experiment.objects.update_or_create(
                project=project_obj,
                consortial_id=sample_id,
                defaults={
                    'name': sample_name,
                    'organism': organism_obj,
                    'project': project_obj,
                    'description': '',
                    'experiment_type': experiment_type_obj,
                    'cell_type': cell_type,
                    'target': protein_target,
                    'slug': sample_name,
                    'public': True,
                },
            )
            # experiments.append(exp_obj)

            ds_obj, ds_created = models.Dataset.objects.update_or_create(
                consortial_id=sample_id,
                experiment=exp_obj,
                defaults={
                    'name': sample_id,
                    'assembly': assembly_obj,
                    'slug': sample_id,
                },
            )
            if forward_bigwig and reverse_bigwig:
                ds_obj.plus_url = forward_bigwig
                ds_obj.minus_url = reverse_bigwig
            else:
                ds_obj.ambiguous_url = unstranded_bigwig
            ds_obj.save()
            datasets.append(ds_obj)

    process_dataset_batch(list(datasets), check_certificate=False)
    for exp_obj in experiments:
        ds_objs = models.Dataset.objects.filter(experiment=exp_obj)
        exp_obj.processed = all([ds_obj.processed for ds_obj in ds_objs])
        exp_obj.save()
