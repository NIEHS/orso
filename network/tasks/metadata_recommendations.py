from celery.decorators import task
from progress.bar import Bar

from network import models


def update_metadata_scores():
    datasets = models.Dataset.objects.all()
    update_dataset_metadata_scores(datasets)


@task
def update_dataset_metadata_scores(datasets):
    '''
    Update or create dataset metadata distance values.
    '''
    bar = Bar('Processing', max=len(datasets))

    onts = []
    onts.append(models.Ontology.objects.get(name='cell_ontology'))
    onts.append(models.Ontology.objects.get(name='cell_line_ontology'))
    onts.append(models.Ontology.objects.get(name='brenda_tissue_ontology'))
    cell_ont_list = [ont.get_ontology_object() for ont in onts]

    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()

    epi_ont = models.Ontology.objects.get(
        name='epigenetic_modification_ontology').get_ontology_object()

    for ds_1 in datasets:
        for ds_2 in models.Dataset.objects.filter(
            assembly=ds_1.assembly,
            experiment__experiment_type=ds_1.experiment.experiment_type,
        ):

            if ds_1 != ds_2:

                exp_1 = models.Experiment.objects.get(dataset=ds_1)
                exp_2 = models.Experiment.objects.get(dataset=ds_2)

                total_sim = 0

                cell_ont_sims = []
                for ont_obj in cell_ont_list:
                    sim = ont_obj.get_word_similarity(
                        exp_1.cell_type, exp_2.cell_type, metric='lin')
                    if sim:
                        cell_ont_sims.append(sim)
                if cell_ont_sims:
                    total_sim += max(cell_ont_sims)

                gene_ont_sim = gene_ont.get_word_similarity(
                    exp_1.target, exp_2.target, metric='jaccard',
                    weighting='information_content')
                if gene_ont_sim:
                    total_sim += gene_ont_sim

                epi_ont_sim = epi_ont.get_word_similarity(
                    exp_1.target, exp_2.target, metric='lin')
                if epi_ont_sim:
                    total_sim += epi_ont_sim

                models.DatasetMetadataDistance.objects.update_or_create(
                    dataset_1=ds_1,
                    dataset_2=ds_2,
                    defaults={
                        'distance': total_sim,
                    },
                )

        bar.next()

    bar.finish()


@task
def update_experiment_metadata_scores(experiments):
    '''
    Update or create experiment metadata distance values.
    '''
    bar = Bar('Processing', max=len(experiments))

    onts = []
    onts.append(models.Ontology.objects.get(name='cell_ontology'))
    onts.append(models.Ontology.objects.get(name='cell_line_ontology'))
    onts.append(models.Ontology.objects.get(name='brenda_tissue_ontology'))
    cell_ont_list = [ont.get_ontology_object() for ont in onts]

    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()

    epi_ont = models.Ontology.objects.get(
        name='epigenetic_modification_ontology').get_ontology_object()

    for exp_1 in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp_1)
        for exp_2 in models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp_1.experiment_type,
        ):

            if exp_1 != exp_2:

                total_sim = 0

                cell_ont_sims = []
                for ont_obj in cell_ont_list:
                    sim = ont_obj.get_word_similarity(
                        exp_1.cell_type, exp_2.cell_type, metric='lin')
                    if sim:
                        cell_ont_sims.append(sim)
                if cell_ont_sims:
                    total_sim += max(cell_ont_sims)

                gene_ont_sim = gene_ont.get_word_similarity(
                    exp_1.target, exp_2.target, metric='jaccard',
                    weighting='information_content')
                if gene_ont_sim:
                    total_sim += gene_ont_sim

                epi_ont_sim = epi_ont.get_word_similarity(
                    exp_1.target, exp_2.target, metric='lin')
                if epi_ont_sim:
                    total_sim += epi_ont_sim

                models.ExperimentMetadataDistance.objects.update_or_create(
                    experiment_1=exp_1,
                    experiment_2=exp_2,
                    defaults={
                        'distance': total_sim,
                    },
                )

        bar.next()

    bar.finish()
