import math
import re
import string
from collections import defaultdict

from fuzzywuzzy import fuzz

tag_prefixes = ['eGFP-', 'FLAG-', 'HA-']


def standardize_word(word):
    '''
    Standardize the word by removing punctuation and making lowercase.
    '''
    for p in string.punctuation:
        word = word.replace(p, '')
    word = ' '.join(word.split()).lower()
    return word


class Ontology:
    '''
    Ontology object containing a name-to-term dictionary and fastsemsim SemSim
    object for calculating pairwise distance.
    '''
    def __init__(self, obo_path, ac_path, ontology_type=None):
        self.obo_path = obo_path
        self.ac_path = ac_path
        self.ontology_type = ontology_type

        self.parse_corpus()
        self.parse_ontology()
        self.determine_information_content()
        self.determine_term_depths()

        self.word_to_terms_cache = dict()
        self.words_to_sim_cache = dict()

    def parse_corpus(self):
        '''
        Parse object names from the AC file (specified by fastsemsim
        documentation). Standardize object names.
        '''
        self.object_to_terms = defaultdict(set)
        self.term_to_objects = defaultdict(set)

        with open(self.ac_path) as f:
            for line in f:
                name, term = line.strip().split('\t')

                if self.ontology_type in ['CLO', 'CL', 'BTO']:
                    name = re.sub(' (cell|cells)$', '', name)

                name = standardize_word(name)
                self.object_to_terms[name].add(term)
                self.term_to_objects[term].add(name)

    def parse_ontology(self):
        '''
        Parse the OBO file to find the name for each term.
        '''
        self.terms = set()
        self.term_to_name = dict()
        self.term_to_parents = defaultdict(set)
        self.term_to_children = defaultdict(set)

        term = None
        name = None
        parents = set()
        skip = True

        with open(self.obo_path) as f:
            for line in f:

                if line.strip() == '[Term]':
                    term = None
                    name = None
                    parents = set()
                    skip = False

                elif line.strip() == '[Typedef]':
                    term = None
                    name = None
                    parents = set()
                    skip = True

                else:
                    if line.startswith('id: '):
                        term = line.strip().split('id: ')[1]
                    elif line.startswith('name: '):
                        try:
                            name = line.strip().split('name: ')[1]
                        except IndexError:
                            print('Empty name field: \"{}\"'.format(
                                  line.strip()))
                    elif line.startswith('is_a:'):
                        parent = (line.strip()
                                  .split('is_a: ')[1]
                                  .split('!')[0].strip())
                        parents.add(parent)
                    elif line.startswith('relationship: part_of'):
                        parent = (line.strip()
                                  .split('relationship: part_of ')[1]
                                  .split('!')[0].strip())
                        parents.add(parent)
                    elif line.startswith('relationship: develops_from '):
                        parent = (line.strip()
                                  .split('relationship: develops_from ')[1]
                                  .split('!')[0].strip())
                        parents.add(parent)

                if term and not skip:
                    self.terms.add(term)
                    self.term_to_name[term] = name
                    self.term_to_parents[term] |= parents
                    for parent in parents:
                        self.term_to_children[parent].add(term)

    def get_all(self, term, term_to_set):
        '''
        Get all objects for a term given a term-to-object dictionary.
        '''
        _all = set()
        if term in term_to_set:
            for n in term_to_set[term]:
                _all.add(n)
                _all |= self.get_all(n, term_to_set)
        return _all

    def get_all_children(self, term):
        '''
        Get all children for a term.
        '''
        return self.get_all(term, self.term_to_children)

    def get_all_parents(self, term):
        '''
        Get all parents for a term.
        '''
        return self.get_all(term, self.term_to_parents)

    def determine_information_content(self):
        '''
        Find information content for each term.

        IC = 1 - ln(n + 1) / ln(N)

        where n is the number of children and N is the number of terms.
        '''
        self.term_information_content = dict()
        term_num = len(self.terms)
        for term in self.terms:
            children_num = len(self.get_all_children(term))
            self.term_information_content[term] = 1 - \
                math.log(children_num + 1) / math.log(term_num)

    def max_depth(self, term):
        '''
        Recursive function to find the max depth of a term given its
        parents.
        '''
        if not self.term_to_parents[term]:
            return 1
        else:
            depths = []
            for parent in self.term_to_parents[term]:
                depths.append(self.max_depth(parent))
            return 1 + max(depths)

    def determine_term_depths(self):
        '''
        Find depth for each term.
        '''
        self.term_to_depth = dict()
        for term in self.terms:
            self.term_to_depth[term] = self.max_depth(term)

    def get_lowest_common_subsumer(self, term_1, term_2):
        '''
        For two terms, get the lowest common subsumer.

        Returns a list of shared parents with lowest depth.
        '''
        term_set_1 = set([term_1]) | self.get_all_parents(term_1)
        term_set_2 = set([term_2]) | self.get_all_parents(term_2)
        intersection = list(term_set_1 & term_set_2)

        if intersection:
            common_subsumers = [{'term': n, 'depth': self.term_to_depth[n]}
                                for n in intersection]
            lowest_depth = max([n['depth'] for n in common_subsumers])
            return [n['term'] for n in common_subsumers
                    if n['depth'] == lowest_depth]
        else:
            return None

    def resnik(self, term_1, term_2, **kwargs):
        '''
        For two terms, get the Resnik Semantic Similarity.

        Because multiple LCSs are possible, return the maximum value.
        '''
        lcs = self.get_lowest_common_subsumer(term_1, term_2)
        if lcs:
            return max([self.term_information_content[n]
                        for n in lcs])
        else:
            return None

    def lin(self, term_1, term_2, **kwargs):
        '''
        For two terms, get the Lin Semantic Similarity.

        Because multiple LCSs are possible, return the maximum value.
        '''
        ic_1 = self.term_information_content[term_1]
        ic_2 = self.term_information_content[term_2]
        lcs = self.get_lowest_common_subsumer(term_1, term_2)
        if lcs:
            return max([2 * self.term_information_content[n] / (ic_1 + ic_2)
                        for n in lcs])
        else:
            return None

    def jaccard(self, term_list_1, term_list_2, include_parents=False,
                weighting=None, **kwargs):
        '''
        For two term lists, get the Jaccard index.
        '''
        term_set_1 = set(term_list_1)
        term_set_2 = set(term_list_2)

        if include_parents:
            for term in term_list_1:
                term_set_1 |= self.get_all_parents(term)
            for term in term_list_2:
                term_set_2 |= self.get_all_parents(term)

        # Define weighting
        if weighting == 'depth':
            def weight(term):
                return self.term_to_depth[term]
        elif weighting == 'information_content':
            def weight(term):
                return self.term_information_content[term]
        else:
            def weight(term):
                return 1.0

        intersection = 0
        union = 0
        for term in term_set_1 & term_set_2:
            intersection += weight(term)
        for term in term_set_1 | term_set_2:
            union += weight(term)

        return intersection / union

    def get_term_list_similarity(self, term_list_1, term_list_2,
                                 metric='resnik', **kwargs):
        '''
        Given two term lists, return the maximum semantic similarity.
        '''
        def _pairwise(term_list_1, term_list_2, function, **kwargs):
            sims = []
            for term_1 in term_list_1:
                for term_2 in term_list_2:
                    sims.append(function(term_1, term_2, **kwargs))
            sims = [n for n in sims if n is not None]
            if sims:
                return max(sims)
            else:
                return None

        if metric == 'resnik':
            return _pairwise(term_list_1, term_list_2, self.resnik, **kwargs)

        elif metric == 'lin':
            return _pairwise(term_list_1, term_list_2, self.lin, **kwargs)

        elif metric == 'jaccard':
            return self.jaccard(term_list_1, term_list_2, **kwargs)

    def get_word_similarity(self, word_1, word_2, metric='resnik', **kwargs):
        '''
        For two words, find the similarity of associated terms.
        '''
        _word_1, _word_2 = sorted([word_1, word_2])

        cache_key = (_word_1, _word_2, metric, frozenset(kwargs.items()))

        if cache_key in self.words_to_sim_cache:
            return self.words_to_sim_cache[cache_key]
        else:

            terms_1 = self.get_terms(_word_1, **kwargs)
            terms_2 = self.get_terms(_word_2, **kwargs)

            if terms_1 and terms_2:
                sim = self.get_term_list_similarity(
                    terms_1, terms_2, metric, **kwargs)
            else:
                sim = None

            self.words_to_sim_cache[cache_key] = sim
            return sim

    def _check_type(self, term):
        '''
        Check if term is consistent with ontology type.
        '''
        if self.ontology_type:
            try:
                term_prefix, term_number = term.split(':')[:2]
            except ValueError:
                print(term)
                raise
            if term_prefix == self.ontology_type:
                return True
            else:
                return False
        else:
            return True

    def get_terms(self, input_word, sm_function=fuzz.ratio, sm_threshold=60,
                  **kwargs):
        '''
        Get ontology terms associated with the input word.
        '''
        cache_key = (input_word, sm_function.__name__, sm_threshold,
                     frozenset(kwargs.items()))

        if cache_key in self.word_to_terms_cache:
            return self.word_to_terms_cache[cache_key]

        word = input_word  # word to be modified

        if self.ontology_type == 'GO':
            for prefix in tag_prefixes:
                word = re.sub('^{}'.format(prefix), '', word)
        elif self.ontology_type in ['CLO', 'CL', 'BTO']:
            word = re.sub(' (cell|cells)$', '', word)
        word = standardize_word(word)

        terms = []

        if word in self.object_to_terms:
            for term in self.object_to_terms[word]:
                if self._check_type(term):
                    terms.append(term)

        if not terms and self.ontology_type != 'GO':

            for key, values in self.object_to_terms.items():
                sim = sm_function(word, key)
                if sim > sm_threshold:
                    for term in values:
                        if self._check_type(term):
                            terms.append({
                                'term': term,
                                'ratio': sim,
                                'depth': self.term_to_depth[term],
                            })

            if terms:
                max_ratio = sorted(
                    terms, key=lambda x: -x['ratio'])[0]['ratio']
                terms_w_max_ratio = \
                    [t for t in terms if t['ratio'] == max_ratio]
                max_depth = sorted(
                    terms_w_max_ratio, key=lambda x: -x['depth'])[0]['depth']
                final_terms = \
                    [t for t in terms_w_max_ratio if t['depth'] == max_depth]
                terms = [t['term'] for t in final_terms]

        self.word_to_terms_cache[cache_key] = terms
        return terms
