from network import models


def update_dataset_data_scores(datasets):
    # Get relevant datasets
    relevant_assemblies = models.Assembly.objects.filter(
        dataset__in=datasets)
    relevant_experiment_types = models.ExperimentType.objects.filter(
        experiment__dataset__in=datasets)
    other_datasets = models.Dataset.objects.filter(
        assembly__in=relevant_assemblies,
        experiment__experiment_type__in=relevant_experiment_types,
    )

    for ds_1 in datasets:

        assembly_1 = ds_1.assembly
        exp_type_1 = ds_1.experiment.experiment_type

        relevant_regions = exp_type_1.relevant_regions

        try:
            pca = models.PCA.objects.get(
                locus_group__assembly=assembly_1,
                experiment_type=exp_type_1,
                locus_group__group_type=relevant_regions,
            )
        except models.PCA.DoesNotExist:
            pass
        else:

            vec_1 = models.PCATransformedValues.objects.get(
                dataset=ds_1, pca=pca).transformed_values

            for ds_2 in other_datasets:

                assembly_2 = ds_2.assembly
                exp_type_2 = ds_2.experiment.experiment_type

                if all([
                    ds_1 != ds_2,
                    assembly_1 == assembly_2,
                    exp_type_1 == exp_type_2,
                ]):

                    vec_2 = models.PCATransformedValues.objects.get(
                        dataset=ds_2, pca=pca).transformed_values
                    vec = pca.neural_network_scaler.transform([vec_1 + vec_2])
                    sim = pca.neural_network.predict(vec)[0]

                    models.DatasetDataDistance.objects.update_or_create(
                        dataset_1=ds_1,
                        dataset_2=ds_2,
                        defaults={
                            'distance': sim,
                        },
                    )
