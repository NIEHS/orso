from scipy.spatial.distance import euclidean, mahalanobis
from sklearn.metrics.pairwise import cosine_distances

from analysis import transform
from network import models


def score_datasets_by_pca_distance(dataset_1, dataset_2):
    '''
    Considering PCATransformedValues, find the distance between datasets by
    transformed values.
    '''
    if dataset_1.assembly != dataset_2.assembly:
        raise ValueError('Datasets do not share genome assemblies.')
    if (dataset_1.experiment.experiment_type !=
            dataset_2.experiment.experiment_type):
        raise ValueError('Datasets do not share experiment type')

    relevant_regions = dataset_1.experiment.experiment_type.relevant_regions
    transformed_1 = models.PCATransformedValues.objects.get(
        dataset=dataset_1, pca__locus_group__group_type=relevant_regions)
    transformed_2 = models.PCATransformedValues.objects.get(
        dataset=dataset_2, pca__locus_group__group_type=relevant_regions)

    if transformed_1.pca != transformed_2.pca:
        raise ValueError('Transformed values do not share PCA object.')

    pca = transformed_1.pca
    inverse_covariation_matrix = pca.inverse_covariation_matrix

    if inverse_covariation_matrix is not None:
        return mahalanobis(
            transformed_1.transformed_values,
            transformed_2.transformed_values,
            inverse_covariation_matrix,
        )
    else:
        return euclidean(
            transformed_1.transformed_values,
            transformed_2.transformed_values,
        )


def score_experiments_by_pca_distance(experiment_1, experiment_2):
    '''
    Considering PCATransformedValues, find the minimum distance between
    experiments by transformed values.
    '''
    if experiment_1.experiment_type != experiment_2.experiment_type:
        raise ValueError('Experiments do not share experiment type.')

    datasets_1 = models.Dataset.objects.filter(experiment=experiment_1)
    datasets_2 = models.Dataset.objects.filter(experiment=experiment_2)

    distances = []
    for ds_1 in datasets_1:
        for ds_2 in datasets_2:
            if ds_1.assembly == ds_2.assembly:
                distances.append(
                    score_datasets_by_pca_distance(ds_1, ds_2))

    if distances:
        return min(distances)
    else:
        raise ValueError(
            'Experiment datasets do not share assembly/annotation.')


def score_datasets_by_tfidf(dataset_1, dataset_2):
    '''
    Considering the metadata of dataset_1 and dataset_2, find the
    distance considering the appropriate TF/IDF matrices.
    '''
    if dataset_1.assembly != dataset_2.assembly:
        raise ValueError('Datasets do not share genome assemblies.')
    if (dataset_1.experiment.experiment_type !=
            dataset_2.experiment.experiment_type):
        raise ValueError('Datasets do not share experiment type')

    vector_1 = transform.dataset_to_tfidf_vector(dataset_1)
    vector_2 = transform.dataset_to_tfidf_vector(dataset_2)

    return cosine_distances(vector_1, vector_2)[0][0]


def score_experiments_by_tfidf(experiment_1, experiment_2):
    '''
    Considering the metadata of experiment_1 and experiment_2, find the
    distance considering the appropriate TF/IDF matrices.

    If the experiments share more than one annotation, it returns the maximum
    score across annotations and TF/IDF matrices.
    '''
    if experiment_1.experiment_type != experiment_2.experiment_type:
        raise ValueError('Experiments do not share experiment type.')

    annotation_set_1 = models.Annotation.objects.filter(
        assembly__dataset__experiment=experiment_1)
    annotation_set_2 = models.Annotation.objects.filter(
        assembly__dataset__experiment=experiment_2)

    distances = []
    for annotation in annotation_set_1 & annotation_set_2:
        vector_1 = transform.experiment_to_tfidf_vector(
            experiment_1, annotation)
        vector_2 = transform.experiment_to_tfidf_vector(
            experiment_2, annotation)
        distances.append(cosine_distances(vector_1, vector_2)[0][0])

    return min(distances)
