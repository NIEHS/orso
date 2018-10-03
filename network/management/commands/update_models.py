from django.core.management.base import BaseCommand

from network import models
from network.management.commands.update_dendrogram import \
    call_update_dendrogram
from network.management.commands.update_pca import call_update_pca
from network.tasks.utils import run_tasks
from network.tasks.analysis.coverage import _set_selected_transcripts_for_genes
from network.tasks.analysis.mlp import \
    fit_neural_network, predict_dataset_fields
from network.tasks.analysis.network import update_organism_network
from network.tasks.analysis.normalization import \
    _normalize_dataset_intersections
from network.tasks.processing import \
    update_or_create_feature_attributes, update_or_create_feature_values
from network.tasks.recommendations import \
    update_experiment_list_recommendations
from network.tasks.similarity import \
    update_dataset_predicted_similarities, \
    update_experiment_metadata_similarities


def update_dendrograms(**kwargs):

    print('Updating dendrograms...')

    tasks = []

    for org in models.Organism.objects.all():
        for exp_type in models.ExperimentType.objects.all():

            experiments = models.Experiment.objects.filter(
                organism=org, experiment_type=exp_type)

            if experiments.exists():
                tasks.append(
                    call_update_dendrogram.si(org.pk, exp_type.pk))

                for my_user in models.MyUser.objects.all():
                    if experiments.filter(owners=my_user).exists():
                        tasks.append(call_update_dendrogram.si(
                            org.pk, exp_type.pk, my_user_pk=my_user.pk))

    run_tasks(tasks, group_async=True)

    print('Done.')


def update_feature_attributes(**kwargs):

    print('Updating feature attributes...')

    tasks = []

    for locus_group in models.LocusGroup.objects.all():
        for experiment_type in models.ExperimentType.objects.all():
            if models.Dataset.objects.filter(
                assembly=locus_group.assembly,
                experiment__experiment_type=experiment_type,
                experiment__project__isnull=False,
            ).exists():
                tasks.append(update_or_create_feature_attributes.si(
                    locus_group.pk, experiment_type.pk))

    run_tasks(tasks, group_async=True)

    print('Done.')


def update_feature_values(**kwargs):

    print('Updating feature values...')

    tasks = []

    for dataset in models.Dataset.objects.all():
        for locus_group in models.LocusGroup.objects.filter(
                assembly=dataset.assembly):
            tasks.append(update_or_create_feature_values.si(
                dataset.pk, locus_group.pk))

    run_tasks(tasks, group_async=True)

    print('Done.')


def update_mlps(**kwargs):

    print('Updating MLPs...')

    tasks = []

    for lg in models.LocusGroup.objects.all():
        for exp_type in models.ExperimentType.objects.all():
            if models.Dataset.objects.filter(
                assembly=lg.assembly,
                experiment__experiment_type=exp_type,
                experiment__project__name='ENCODE',
                experiment__processed=True,
                experiment__revoked=False,
                processed=True,
                revoked=False,
            ).count() >= 20:

                nn = models.NeuralNetwork.objects.get_or_create(
                    locus_group=lg,
                    experiment_type=exp_type,
                    metadata_field='cell_type',
                )[0]
                tasks.append(fit_neural_network.si(nn.pk))

                nn = models.NeuralNetwork.objects.get_or_create(
                    locus_group=lg,
                    experiment_type=exp_type,
                    metadata_field='target',
                )[0]
                tasks.append(fit_neural_network.si(nn.pk))

    run_tasks(tasks, group_async=True)

    print('Done.')


def update_networks(**kwargs):

    print('Updating network graphs...')

    tasks = []

    for org in models.Organism.objects.all():
        for exp_type in models.ExperimentType.objects.all():

            experiments = models.Experiment.objects.filter(
                dataset__assembly__organism=org,
                experiment_type=exp_type,
            )

            if experiments.exists():
                tasks.append(
                    update_organism_network.si(org.pk, exp_type.pk))

                for my_user in models.MyUser.objects.all():
                    if experiments.filter(owners=my_user).exists():
                        tasks.append(update_organism_network.si(
                            org.pk, exp_type.pk, my_user_pk=my_user.pk))

    run_tasks(tasks, group_async=True)

    print('Done.')


def update_normalized_read_coverage(**kwargs):

    print('Updating normalized read coverage...')

    tasks = []

    dataset_intersections = models.DatasetIntersectionJson.objects.filter(
        dataset__processed=True).distinct()
    for dij in dataset_intersections:
        tasks.append(_normalize_dataset_intersections.si(dij.pk))

    run_tasks(tasks, group_async=True)

    print('Done.')


def update_pcas(**kwargs):
    print('Updating PCAs...')

    tasks = []

    for lg in models.LocusGroup.objects.all():
        for exp_type in models.ExperimentType.objects.all():
            if models.Dataset.objects.filter(
                assembly=lg.assembly,
                experiment__experiment_type=exp_type,
            ).count() >= 3:  # Verify that there are at least 3 datasets
                tasks.append(call_update_pca.si(lg.pk, exp_type.pk))

    run_tasks(tasks, group_async=True)

    print('Done.')


def update_predictions(**kwargs):

    print('Updating dataset predictions...')

    tasks = []
    for ds in models.Dataset.objects.all():
        tasks.append(predict_dataset_fields.si(ds.pk))

    run_tasks(tasks, group_async=True)

    print('Done.')


def update_recommendations(**kwargs):

    print('Updating recommendations...')

    experiments = models.Experiment.objects.all()
    update_experiment_list_recommendations(
        experiments, sim_types=['metadata', 'primary'], group_async=True)

    print('Done.')


def update_similarities(**kwargs):

    print('Updating similarities...')

    datasets = models.Dataset.objects.all()
    experiments = models.Experiment.objects.all()

    update_dataset_predicted_similarities(datasets, group_async=True)
    update_experiment_metadata_similarities(experiments, group_async=True)

    print('Done.')


def update_transcript_selections(**kwargs):

    print('Updating transcript selections...')

    tasks = []

    annotations = models.Annotation.objects.filter(
        gene__isnull=False).distinct()
    for annotation in annotations:
        tasks.append(_set_selected_transcripts_for_genes.si(annotation.pk))

    run_tasks(tasks, group_async=True)

    print('Done.')


# Order is important here; if multiple flags are used, they will be executed
# in this order.
COMMANDS = [
    ('--normalized_read_coverage', 'Update normalized read coverage',
        update_normalized_read_coverage),
    ('--transcript_selections', 'Update transcript selections for genes',
        update_transcript_selections),

    ('--feature_attributes', 'Update features (gene/enhancer) attributes',
        update_feature_attributes),
    ('--feature_values', 'Update features (gene/enhancer) attributes',
        update_feature_values),

    ('--pcas', 'Update PCA models', update_pcas),
    ('--mlps', 'Update MLP neural networks', update_mlps),

    ('--predictions', 'Update dataset predictions', update_predictions),
    ('--similarities', 'Update similarities', update_similarities),
    ('--recommendations', 'Update recommendations', update_recommendations),

    ('--dendrograms', 'Update dendrograms', update_dendrograms),
    ('--networks', 'Update network graphs', update_networks),
]


class Command(BaseCommand):
    help = '''
        Update network models.
    '''

    def add_arguments(self, parser):
        for flag, help_string, func in COMMANDS:
            parser.add_argument(flag, action='store_true', help=help_string)

    def handle(self, *args, **options):

        if all([not options[x[0].split('--')[1]] for x in COMMANDS]):
            print('No models selected for update.  Try "--help" for a list of '
                  'appropriate flags.')

        else:

            for flag, help_string, func in COMMANDS:
                if options[flag.split('--')[1]]:
                    func(threads=options['threads'])
