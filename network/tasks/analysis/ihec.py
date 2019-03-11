import json
import re
from subprocess import call, DEVNULL

from network import models
from network.tasks.analysis.utils import download_dataset_bigwigs
from network.tasks.processing import process_experiment
from network.tasks.utils import run_tasks

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

    processed = 0

    # Iterate through 'datasets'
    for dataset_name, info in list(ihec_dict['datasets'].items()):

        skip = False

        project_name = info['ihec_data_portal']['publishing_group']
        cell_type = info['ihec_data_portal']['cell_type']
        assay_type = info['ihec_data_portal']['assay']

        sample_id = info['sample_id']

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

            if protein_target:
                exp_obj = models.Experiment.objects.get_or_create(
                    name=sample_name,
                    cell_type=cell_type,
                    consortial_id=sample_id,
                    description='',
                    experiment_type=experiment_type_obj,
                    organism=organism_obj,
                    project=project_obj,
                    public=True,
                    slug=sample_name,
                    target=protein_target,
                )[0]
            else:
                exp_obj = models.Experiment.objects.get_or_create(
                    name=sample_name,
                    cell_type=cell_type,
                    consortial_id=sample_id,
                    description='',
                    experiment_type=experiment_type_obj,
                    organism=organism_obj,
                    project=project_obj,
                    public=True,
                    slug=sample_name,
                )[0]

            if exp_obj.processed is True:

                processed += 1

            else:

                experiments.append(exp_obj)

                ds_obj = models.Dataset.objects.get_or_create(
                    assembly=assembly_obj,
                    consortial_id=sample_id,
                    experiment=exp_obj,
                    name=sample_id,
                    slug=sample_id,
                )[0]
                if forward_bigwig and reverse_bigwig:
                    ds_obj.plus_url = forward_bigwig
                    ds_obj.minus_url = reverse_bigwig
                else:
                    ds_obj.ambiguous_url = unstranded_bigwig
                ds_obj.save()

    print('{} experiments already processed.'.format(str(processed)))
    print('{} experiments queued for processing'.format(
        str(len(experiments))))

    tasks = []
    for experiment in experiments:
        tasks.append(process_experiment.si(
            experiment.pk, update_recs_and_sims=False))
    run_tasks(tasks, group_async=True)
