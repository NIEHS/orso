import string

from network import models

EXPERIMENT_DESCRIPTION_FIELDS = [
    'assay_slims',
    'assay_synonyms',
    'assay_term_name',
    'assay_title',
    'biosample_summary',
    'biosample_synonyms',
    'biosample_term_name',
    'biosample_type',
    'category_slims',
    'objective_slims',
    'organ_slims',
    'target',
    'system_slims',
]


def pca_transform_intersections(dataset):
    '''
    Transform the intersection values for a dataset by the associated PCA.
    '''
    exp_type = models.ExperimentType.objects.get(
        experiment__dataset=dataset,
    )
    pca = models.PCA.objects.get(
        experiment_type=exp_type,
        locus_group__group_type=exp_type.relevant_regions,
        locus_group__assembly=dataset.assembly,
    )

    order = models.PCALocusOrder.objects.filter(pca=pca).order_by('order')
    loci = [x.locus for x in order]

    intersection_values = []
    for locus in loci:
        intersection_values.append(
            models.DatasetIntersection.objects.get(
                dataset=dataset, locus=locus).normalized_value
        )

    return pca.pca.transform([intersection_values])[0]


def experiment_description_to_list(experiment):
    '''
    Given an experiment object, transform the metadata into a term list.
    '''
    description = experiment.description
    description = description.translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    description_list = description.split()
    for field in EXPERIMENT_DESCRIPTION_FIELDS:
        description_list = [x for x in description_list if x != field]

    if experiment.cell_type:
        description_list.append(experiment.cell_type)
    if experiment.target:
        description_list.append(experiment.target)

    return description_list


def experiment_to_tfidf_vector(experiment, assembly):
    '''
    Given the experiment object, transform the metadata into a TF/IDF vector.
    '''
    tfidf_vectorizer = models.TfidfVectorizer.objects.get(
        assembly=assembly,
        experiment_type=experiment.experiment_type,
    )
    term_list = experiment_description_to_list(experiment)

    return tfidf_vectorizer.tfidf_vectorizer.transform([' '.join(term_list)])


def dataset_to_tfidf_vector(dataset):
    '''
    Given the dataset object, transform the metadata into a TF/IDF vector.
    '''
    assembly = dataset.assembly
    experiment = dataset.experiment

    return experiment_to_tfidf_vector(experiment, assembly)
