import json
import math
from collections import defaultdict

from network import models


def semantic_idf(word_set_1, word_set_2, idf, sim_function):
    '''
    Find the SemSimIDF value for two experiments considering IDF values and a
    semantic similarity function.

    The semantic similarity function must take in two word sets as positional
    arguments.

    If the IDF dict is not specified, use the get method to grab an IDF object
    and load the associated JSON.
    '''
    # if not idf:
    #     if experiment_1.experiment_type == experiment_2.experiment_type and \
    #             experiment_1.assembly == experiment_2.assembly:
    #         idf = json.loads(models.IDF.objects.get(
    #             assembly=experiment_1.assembly,
    #             experiment_type=experiment_1.experiment_type,
    #         ).idf)
    #     else:
    #         raise Exception('Assembly and experiment type are not consistent.')

    def _get_semantic_idf_term(word_set_1, word_set_2):
        total_numerator = 0
        total_denominator = 0

        for word_1 in word_set_1:
            if word_1 in idf:
                sim_values = []
                for word_2 in word_set_2:
                    val = sim_function(word_1, word_2)
                    if val:
                        sim_values.append(val)
                if sim_values:
                    total_numerator += max(sim_values) * idf[word_1]
                    total_denominator += idf[word_1]

        if total_denominator == 0:
            return 0
        else:
            return total_numerator / total_denominator

    # word_set_1 = experiment_1.get_word_set()
    # word_set_2 = experiment_2.get_word_set()

    term_1 = _get_semantic_idf_term(word_set_1, word_set_2)
    term_2 = _get_semantic_idf_term(word_set_2, word_set_1)

    return (term_1 + term_2) / 2


def generate_idf(assembly, experiment_type):
    '''
    Considering Assembly and ExperimentType objects, collect Experiment objects
    and generate an IDF.
    '''
    experiments = models.Experiment.objects.filter(
        dataset__assembly=assembly,
        experiment_type=experiment_type,
    )
    return experiments_to_idf(experiments)


def experiments_to_idf(experiments):
    '''
    Take a list of Experiment objects and generate an IDF dict.
    '''
    word_sets = []
    for experiment in experiments:
        word_sets.append(experiment.get_word_set())
    return word_sets_to_idf_dict(word_sets)


def word_sets_to_idf_dict(word_sets):
    '''
    Take a list of word sets and generate an IDF dict.
    '''
    word_counts = defaultdict(int)

    for word_set in word_sets:
        for word in word_set:
            word_counts[word] += 1

    idf_dict = dict()
    for word, count in word_counts.items():
        idf_dict[word] = math.log10(len(word_sets) / count)

    return idf_dict
