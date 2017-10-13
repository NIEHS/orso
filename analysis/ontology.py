import difflib
import string

from fastsemsim import SemSim
from fastsemsim.Ontology import AnnotationCorpus, ontologies


def standardize_word(word):
    '''
    Standardize the word by removing spaces and punctuation and making
    lowercase.
    '''
    for p in string.punctuation:
        word = word.replace(p, '')
    word = ''.join(word.split()).lower()
    return word


def get_max_score_from_ontology_list(word_1, word_2, ontology_list):
    '''
    Given two words and a list of ontology objects, find the max semantic
    similarity score. If all reported scores are None, return None.
    '''
    scores = []
    for ontology in ontology_list:
        scores.append(ontology.get_similarity(word_1, word_2))
    scores = [x for x in scores if x is not None]
    if scores:
        return max(scores)
    else:
        return None


class Ontology:
    '''
    Ontology object containing a name-to-term dictionary and fastsemsim SemSim
    object for calculating pairwise distance.
    '''
    def __init__(self, obo_path, ac_path, ontology_type):
        self.obo_path = obo_path
        self.ac_path = ac_path
        self.ontology_type = ontology_type

        self.sim_cache = dict()
        self.word_to_terms_cache = dict()

        self.generate_object_to_term()
        self.generate_semsim()

    def generate_object_to_term(self):
        '''
        Parse object names from the AC file (specified by fastsemsim
        documentation). Standardize object names.
        '''
        self.object_to_term = dict()

        with open(self.ac_path) as f:
            for line in f:
                name, term = line.strip().split('\t')
                self.object_to_term[standardize_word(name)] = term

    def generate_semsim(self):
        '''
        Generate a SemSim object (fastsemsim) from OBO and AC files.
        '''
        ont_obj = ontologies.load(
            self.obo_path,
            source_type='obo',
            ontology_type=self.ontology_type,
            parameters={'ignore': None},
        )
        ac_obj = AnnotationCorpus.AnnotationCorpus(ont_obj)
        ac_obj.parse(
            self.ac_path,
            'plain',
        )
        tss_class = SemSim.select_term_SemSim('Lin')
        self.ss = tss_class(
            ont_obj,
            ac_obj,
            SemSim.SemSimUtils(ont_obj, ac_obj),
        )

    def get_similarity(self, word_1, word_2):
        '''
        Find the maximum semantic similarity given two words.
        '''
        word_pair = tuple(sorted([word_1, word_2]))
        if word_pair in self.sim_cache:
            return self.sim_cache[word_pair]

        term_list_1 = self.get_terms(word_1)
        term_list_2 = self.get_terms(word_2)

        sim = self.get_term_list_similarity(term_list_1, term_list_2)
        self.sim_cache[word_pair] = sim
        return sim

    def get_terms(self, word):
        '''
        Get ontology terms associated with the input word.
        '''
        if word in self.word_to_terms_cache:
            return self.word_to_terms_cache[word]

        _dict = self.object_to_term
        std_word = standardize_word(word)

        if std_word in _dict:
            terms = [_dict[std_word]]
        else:
            terms = []
            for key, term in _dict.items():
                if std_word in key:
                    _ratio = difflib.SequenceMatcher(
                        None, key.lower(), std_word.lower()).ratio()
                    if _ratio >= 0.6:
                        terms.append(term)
        self.word_to_terms_cache[word] = terms
        return terms

    def get_term_list_similarity(self, term_list_1, term_list_2):
        '''
        Given two term lists, return the maximum semantic similarity.
        '''
        sims = []
        for term_1 in term_list_1:
            for term_2 in term_list_2:
                sim = self.ss.SemSim(
                    term_1,
                    term_2,
                    None,
                )
                sims.append(sim)
        sims = [x for x in sims if x is not None]
        if sims:
            return max(sims)
        else:
            return None

    def filter_word_set_for_ontolgy(self, word_set):
        '''
        For a word set, filter by occurrence in the name-to-term dictionary.
        '''
        output_word_set = set()
        for word in word_set:
            if word in self.object_to_term:
                output_word_set.add(word)
            else:
                for name in self.object_to_term.keys():
                    if word in name:
                        output_word_set.add(word)
        return output_word_set
