from django.db.models import Q

from network import models
from analysis.encode import EncodeProject
from network.tasks.process_datasets import process_dataset_batch


def revoke_missing_experiments(encode_obj, project):
    query = Q()
    for experiment in encode_obj.experiments:
        query |= Q(consortial_id=experiment.id)
    query_set = (models.Experiment.objects
                                  .filter(project=project)
                                  .exclude(query))
    for experiment in query_set:
        experiment.revoked = True
        experiment.save()
    print('{} experiments missing from query. Revoked!!'.format(
        str(query_set.count())))


def revoke_missing_datasets(encode_obj, project):
    query = Q()
    for experiment in encode_obj.experiments:
        for dataset in experiment.datasets:
            for attr in ['ambiguous', 'plus', 'minus']:
                try:
                    query |= Q(
                        consortial_id__contains=getattr(dataset, attr).id)
                except AttributeError:
                    pass
    query_set = (models.Dataset.objects
                               .filter(experiment__project=project)
                               .exclude(query))
    for dataset in query_set:
        dataset.revoked = True
        dataset.save()
    print('{} datasets missing from query. Revoked!!'.format(
        str(query_set.count())))


def revoke_experiments_with_revoked_datasets(encode_obj, project):
    query_set = (models.Experiment.objects
                                  .filter(revoked=False, project=project)
                                  .exclude(dataset__revoked=False))
    for experiment in query_set:
        experiment.revoked = True
        experiment.save()
    print('{} experiments with only revoked datasets. Revoked!!'.format(
        str(query_set.count())))


def add_or_update_encode():

    project = models.Project.objects.get_or_create(
        name='ENCODE',
    )[0]

    encode = EncodeProject()
    print('{} experiments found in ENCODE!!'.format(len(encode.experiments)))

    datasets_to_process = set()

    # Create experiment and dataset objects; process datasets
    for experiment in encode.experiments:

        try:
            experiment_type_obj = models.ExperimentType.objects.get(
                name=experiment.experiment_type)
        except models.ExperimentType.DoesNotExist:
            experiment_type_obj = models.ExperimentType.objects.create(
                name=experiment.experiment_type,
                short_name=experiment.short_experiment_type,
                relevant_regions='genebody',
            )

        # Update or create experiment object
        exp_obj, exp_created = models.Experiment.objects.update_or_create(
            project=project,
            consortial_id=experiment.id,
            defaults={
                'name': experiment.name,
                'project': project,
                'description': experiment.description,
                'experiment_type': experiment_type_obj,
                'cell_type': experiment.cell_type,
                'slug': experiment.name,
            },
        )
        if experiment.target:
            exp_obj.target = experiment.target
            exp_obj.save()

        for dataset in experiment.datasets:

            # Get assembly object
            try:
                assembly_obj = \
                    models.Assembly.objects.get(name=dataset.assembly)
            except models.Assembly.DoesNotExist:
                assembly_obj = None
                print(
                    'Assembly "{}" does not exist for dataset {}. '
                    'Skipping dataset.'.format(dataset.assembly, dataset.name)
                )

            # Add dataset
            if assembly_obj:

                # Update or create dataset
                ds_obj, ds_created = models.Dataset.objects.update_or_create(
                    consortial_id=dataset.id,
                    experiment=exp_obj,
                    defaults={
                        'name': dataset.name,
                        'assembly': assembly_obj,
                        'slug': dataset.id,
                    },
                )

                # Update URLs, if appropriate
                updated_url = False
                if dataset.ambiguous:
                    if ds_obj.ambiguous_url != dataset.ambiguous.url:
                        ds_obj.ambiguous_url = dataset.ambiguous.url
                        updated_url = True
                elif dataset.plus and dataset.minus:
                    if any([
                        ds_obj.plus_url != dataset.plus.url,
                        ds_obj.minus_url != dataset.minus.url,
                    ]):
                        ds_obj.plus_url = dataset.plus.url
                        ds_obj.minus_url = dataset.minus.url
                        updated_url = True
                if updated_url:
                    ds_obj.processed = False
                    ds_obj.save()

                if not ds_obj.processed:
                    datasets_to_process.add(ds_obj)

    print('Processing {} datasets...'.format(len(datasets_to_process)))
    process_dataset_batch(list(datasets_to_process))

    revoke_missing_experiments(encode, project)
    revoke_missing_datasets(encode, project)
    revoke_experiments_with_revoked_datasets(encode, project)
