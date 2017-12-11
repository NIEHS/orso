from celery.decorators import task
from progress.bar import Bar

from network import models


@task
def update_dataset_metadata_scores(datasets):
    '''
    Update or create dataset metadata distance values.
    '''
    updated = set()

    bar_max = 0
    for ds in datasets:
        bar_max += models.Dataset.objects.filter(
            assembly=ds.assembly,
            experiment__experiment_type=ds.experiment.experiment_type,
        ).count()
    bar = Bar('Processing', max=bar_max)

    onts = []
    onts.append(models.Ontology.objects.get(name='cell_ontology'))
    onts.append(models.Ontology.objects.get(name='cell_line_ontology'))
    onts.append(models.Ontology.objects.get(name='brenda_tissue_ontology'))
    cell_ont_list = [ont.get_ontology_object() for ont in onts]

    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()

    for ds_1 in datasets:
        for ds_2 in models.Dataset.objects.filter(
            assembly=ds_1.assembly,
            experiment__experiment_type=ds_1.experiment.experiment_type,
        ):

            _ds_1, _ds_2 = sorted([ds_1, ds_2], key=lambda x: x.pk)
            if all([
                (_ds_1, _ds_2) not in updated,
                _ds_1 != _ds_2,
            ]):

                exp_1 = models.Experiment.objects.get(dataset=_ds_1)
                exp_2 = models.Experiment.objects.get(dataset=_ds_2)

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

                models.DatasetMetadataDistance.objects.update_or_create(
                    dataset_1=_ds_1,
                    dataset_2=_ds_2,
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
    updated = set()

    bar_max = 0
    for exp in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp)
        bar_max += models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp.experiment_type,
        ).count()
    bar = Bar('Processing', max=bar_max)

    onts = []
    onts.append(models.Ontology.objects.get(name='cell_ontology'))
    onts.append(models.Ontology.objects.get(name='cell_line_ontology'))
    onts.append(models.Ontology.objects.get(name='brenda_tissue_ontology'))
    cell_ont_list = [ont.get_ontology_object() for ont in onts]

    gene_ont = models.Ontology.objects.get(
        name='gene_ontology').get_ontology_object()

    for exp_1 in experiments:
        assemblies = models.Assembly.objects.filter(dataset__experiment=exp_1)
        for exp_2 in models.Experiment.objects.filter(
            dataset__assembly__in=assemblies,
            experiment_type=exp_1.experiment_type,
        ):

            _exp_1, _exp_2 = sorted([exp_1, exp_2], key=lambda x: x.pk)
            if all([
                (_exp_1, _exp_2) not in updated,
                _exp_1 != _exp_2,
            ]):

                total_sim = 0

                cell_ont_sims = []
                for ont_obj in cell_ont_list:
                    sim = ont_obj.get_word_similarity(
                        _exp_1.cell_type, _exp_2.cell_type, metric='lin')
                    if sim:
                        cell_ont_sims.append(sim)
                if cell_ont_sims:
                    total_sim += max(cell_ont_sims)

                gene_ont_sim = gene_ont.get_word_similarity(
                    _exp_1.target, _exp_2.target, metric='jaccard',
                    weighting='information_content')
                if gene_ont_sim:
                    total_sim += gene_ont_sim

                models.ExperimentMetadataDistance.objects.update_or_create(
                    experiment_1=_exp_1,
                    experiment_2=_exp_2,
                    defaults={
                        'distance': total_sim,
                    },
                )
                updated.add((_exp_1, _exp_2))

            bar.next()

    bar.finish()
