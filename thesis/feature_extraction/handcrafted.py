# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import numpy as np

from collections import defaultdict
from itertools import chain
from sklearn.feature_extraction import FeatureHasher


_NOT_VALID_DEP_RELATIONS = {'modnorule', 'modnomatch'}


class HandcraftedFeaturesExtractor(object):
    def __init__(self, **features):
        # Main word features
        self._main_token = features.pop('main_token', True)
        self._main_lemma = features.pop('main_lemma', True)
        self._main_tag = features.pop('main_tag', True)
        self._main_morpho = features.pop('main_morpho', True)  # Morphosyntactic info (only for Spanish)

        # Surrounding words features (collocations and bag of words)
        self._window_size = features.pop('windows_size', 5)
        self._window_bow = features.pop('window_bow', True)
        self._window_tokens = features.pop('window_tokens', True)
        self._window_lemmas = features.pop('window_lemmas', True)
        self._window_tags = features.pop('window_tags', True)
        self._surrounding_bigrams = features.pop('surrounding_bigrams', True)
        self._surrounding_trigrams = features.pop('surrounding_trigrams', True)

        # Features from words dependant on the main word (tails of the relation)
        self._inbound_dep_triples = features.pop('inbound_dep_triples', True)

        # Features of the word the main word depends (head of the relation)
        self._outbound_dep_triple = features.pop('outbound_dep_triple', True)

    def featurize_sentence(self, sentence, main_word_index):
        """
        Takes a sentence and creates a feature dictionary
        :type sentence: corpora.parsers.Sentence
        :type main_word_index: int
        :return: Dictionary of features
        """
        features_dict = defaultdict(int)

        # Main word features
        main_word = sentence.get_word_by_index(main_word_index)

        if self._main_token:
            features_dict['main_token'] = main_word.token
        if self._main_lemma:
            features_dict['main_lemma'] = main_word.lemma
        if self._main_tag:
            features_dict['main_tag'] = main_word.tag
        if self._main_morpho:
            for info, value in (info.split('=') for info in main_word.morpho_info):
                features_dict['morpho:%s' % info] = value

        if self._window_size > 0:
            leftest_word_index = max(main_word_index - self._window_size, 1)
            rightest_word_index = min(main_word_index + self._window_size, len(sentence) + 1)

            for window_word_index in chain(range(leftest_word_index, main_word_index),
                                           range(main_word_index+1, rightest_word_index)):
                window_word = sentence.get_word_by_index(window_word_index)
                if self._window_bow:
                    features_dict['bow:%s' % window_word.token] += 1

                relative_position = window_word_index - main_word_index

                if self._window_tokens:
                    features_dict['token%+d' % relative_position] = window_word.token

                if self._window_lemmas:
                    features_dict['lemma%+d' % relative_position] = window_word.lemma

                if self._window_tags:
                    features_dict['tag%+d' % relative_position] = window_word.tag

        if self._surrounding_bigrams and main_word_index > 2:
            features_dict['left_bigram'] = '%s %s' % (
                sentence.get_word_by_index(main_word_index-2).token,
                sentence.get_word_by_index(main_word_index-1).token
            )

        if self._surrounding_bigrams and main_word_index <= len(sentence) - 2:
            features_dict['right_bigram'] = '%s %s' % (
                sentence.get_word_by_index(main_word_index+1).token,
                sentence.get_word_by_index(main_word_index+2).token
            )

        if self._surrounding_trigrams and main_word_index > 3:
            features_dict['left_trigram'] = '%s %s %s' % (
                sentence.get_word_by_index(main_word_index-3).token,
                sentence.get_word_by_index(main_word_index-2).token,
                sentence.get_word_by_index(main_word_index-1).token
            )

        if self._surrounding_trigrams and main_word_index <= len(sentence) - 3:
            features_dict['right_trigram'] = '%s %s %s' % (
                sentence.get_word_by_index(main_word_index+1).token,
                sentence.get_word_by_index(main_word_index+2).token,
                sentence.get_word_by_index(main_word_index+3).token
            )

        if self._inbound_dep_triples:
            for word in (word for word in sentence
                         if word.dep_head == main_word.idx
                         and word.dep_rel not in _NOT_VALID_DEP_RELATIONS):
                features_dict['dep:%s:%s:%s' % (word.token, word.dep_rel, main_word.token)] += 1

        if self._outbound_dep_triple:
            if main_word.dep_rel == 'top':
                features_dict['dep:%s:top' % main_word.token] += 1
            elif main_word.dep_rel not in _NOT_VALID_DEP_RELATIONS:
                dep_word = sentence.get_word_by_index(main_word.dep_head)
                features_dict['dep:%s:%s:%s' % (main_word.token, main_word.dep_rel, dep_word.token)] += 1

        return features_dict


class HandcraftedHashedFeaturesExtractor(HandcraftedFeaturesExtractor):
    def __init__(self, n_features=2**13, dtype=np.float32, non_negative=True, **features):
        super(HandcraftedHashedFeaturesExtractor, self).__init__(**features)
        self._hasher = FeatureHasher(n_features=n_features, dtype=dtype, non_negative=non_negative)

    def featurize_sentence(self, sentence, main_word_index):
        """
        Takes a sentence and creates a feature vector using the FeautureHasher
        :type sentence: corpora.parsers.Sentence
        :type main_word_index: int
        :return: Vector representing the sentence
        """
        features_dict = super(HandcraftedHashedFeaturesExtractor, self).featurize_sentence(sentence, main_word_index)

        return self._hasher.transform([features_dict])
