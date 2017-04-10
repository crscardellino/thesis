# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import numpy as np
import pickle

from collections import namedtuple
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix


_InstanceId = namedtuple('InstanceId', 'corpu file sentence lemma idx')


class CorpusDataset(object):
    def __init__(self, dataset, feature_dict_path=None, word_vector_model=None, dtype=np.float32):
        if feature_dict_path is not None:
            with open(feature_dict_path, 'rb') as f:
                self._features_dict = pickle.load(f)

        if word_vector_model is None:
            self._data = csr_matrix((dataset['data'], dataset['indices'], dataset['indptr']),
                                    shape=dataset['shape'], dtype=dtype)
            self._input_vector_size = self._data.shape[1]
            self._word_vector_model = None
        else:
            self._data = dataset['data']
            self._word_vector_model = word_vector_model

            try:
                self._word_vector_size = self._word_vector_model.vector_size
            except AttributeError:
                self._word_vector_size = next(iter(self._word_vector_model.values())).shape[0]

            self._input_vector_size = self._data.shape[1] * self._word_vector_size

        self.dtype = dtype

    def input_vector_size(self):
        return self._input_vector_size

    def num_examples(self):
        return self._data.shape[0]


class SenseCorpusDataset(CorpusDataset):
    def __init__(self, dataset_path, features_dict_path=None, word_vector_model=None, dtype=np.float32):
        dataset = np.load(dataset_path)
        super(SenseCorpusDataset, self).__init__(dataset, features_dict_path, word_vector_model, dtype)

        if word_vector_model is None:
            self._data = csr_matrix((dataset['data'], dataset['indices'], dataset['indptr']),
                                    shape=dataset['shape'], dtype=dtype)
            self._input_vector_size = self._data.shape[1]
            self._word_vector_model = None
        else:
            self._data = dataset['data']
            self._word_vector_model = word_vector_model

            try:
                self._word_vector_size = self._word_vector_model.vector_size
            except AttributeError:
                self._word_vector_size = next(iter(self._word_vector_model.values())).shape[0]

            self._input_vector_size = self._data.shape[1] * self._word_vector_size

        self._target = dataset['target']

        self._lemmas = dataset['lemmas']
        self._unique_lemmas = np.unique(self._lemmas)
        self._sentences = dataset['sentences']
        self._train_classes = dataset['train_classes']

    def _word_window_to_vector(self, word_window):
        vector = []

        for word in word_window:
            try:
                vector.append(self._word_vector_model[next(t for t in word if t in self._word_vector_model)])
            except StopIteration:
                vector.append(np.zeros(self._word_vector_size, dtype=self.dtype))

        return np.concatenate(vector)

    def data(self, lemma=None):
        data = self._data if lemma is None else self._data[np.where(self._lemmas == lemma)[0], :]

        if self._word_vector_model is not None:
            data = np.array([self._word_window_to_vector(ww) for ww in data])

        return data

    def target(self, lemma=None):
        if lemma is None:
            return self._target
        else:
            return self._target[np.where(self._lemmas == lemma)[0]]

    def traverse_dataset_by_lemma(self):
        for lemma in self._unique_lemmas:
            yield lemma, self.data(lemma), self.target(lemma)

    def output_vector_size(self, lemma=None):
        if lemma is None:
            return self._train_classes.shape[0]
        else:
            return np.array([cls for cls in self._train_classes if lemma == cls.split('.')[1]]).shape[0]

    def num_examples(self, lemma=None):
        return self.data(lemma).shape[0]

    @property
    def num_lemmas(self):
        return self._unique_lemmas.shape[0]


class SenseCorpusDatasets(object):
    def __init__(self, train_dataset_path, test_dataset_path, word_vector_model_path=None, dtype=np.float32):
        try:
            word_vector_model = Word2Vec.load_word2vec_format(word_vector_model_path, binary=True)\
                if word_vector_model_path is not None else None
        except UnicodeDecodeError:
            with open(word_vector_model_path, 'rb') as fvectors:
                word_vector_model = pickle.load(fvectors)

        self.train_dataset = SenseCorpusDataset(train_dataset_path, word_vector_model, dtype)
        self.test_dataset = SenseCorpusDataset(test_dataset_path, word_vector_model, dtype)


class UnlabeledCorpusDataset(CorpusDataset):
    def __init__(self, dataset_path, features_dict_path=None, word_vector_model=None, dtype=np.float32):
        dataset = np.load(dataset_path)
        super(UnlabeledCorpusDataset, self).__init__(dataset, features_dict_path, word_vector_model, dtype)

        self.instances_id = [_InstanceId(*iid.split(':')) for iid in dataset['instances_id']]
