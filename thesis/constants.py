# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from thesis.classification import BaselineClassifier, KerasMultilayerPerceptron


CLASSIFIERS = {
    'baseline': BaselineClassifier,
    'decision_tree': DecisionTreeClassifier,
    'log': LogisticRegression,
    'mlp': KerasMultilayerPerceptron,
    'naive_bayes': MultinomialNB,
    'svm': SVC
}

DEFAULT_FEATURES = {
    'main_token': True, 'main_lemma': True, 'main_tag': True, 'main_morpho': True,
    'window_bow': True, 'window_tokens': True, 'window_lemmas': True, 'window_tags': True,
    'surrounding_bigrams': True, 'surrounding_trigrams': True,
    'inbound_dep_triples': True, 'outbound_dep_triples': True
}

FEATURE_SELECTION = {
    'chi2': chi2,
    'f_classif': f_classif,
    'mutual_info_classif': mutual_info_classif
}

LANGUAGE = {
    'spanish': 'sensem',
    'english': 'semeval'
}

SENSEM_COLUMNS = ('idx', 'token', 'lemma', 'tag', 'morpho_info', 'ner', 'dep_head', 'dep_rel')
SEMEVAL_COLUMNS = ('idx', 'token', 'lemma', 'tag', 'ner', 'dep_head', 'dep_rel')
CORPUS_COLUMNS = {
    'semeval': SEMEVAL_COLUMNS,
    'sensem': SENSEM_COLUMNS
}

RANDOM_SEED = 1234