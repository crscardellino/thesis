# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import numpy as np

from scipy.sparse import csr_matrix


class SenseCorpusDataset(object):
    def __init__(self, dataset_path, dtype=np.float32):
        dataset = np.load(dataset_path)

        self._data = csr_matrix((dataset['data'], dataset['indices'], dataset['indptr']),
                                shape=dataset['shape'], dtype=dtype)
        self._target = dataset['target']

        self._lemmas = dataset['lemmas']
        self._unique_lemmas = np.unique(self._lemmas)
        self._sentences = dataset['sentences']
        self._train_classes = dataset['train_classes']
        self.dtype = dtype

    def data(self, lemma=None):
        if lemma is None:
            return self._data
        else:
            return self._data[np.where(self._lemmas == lemma)[0], :]

    def target(self, lemma=None):
        if lemma is None:
            return self._target
        else:
            return self._target[np.where(self._lemmas == lemma)[0]]

    def traverse_dataset_by_lemma(self):
        for lemma in self._unique_lemmas:
            yield lemma, self.data(lemma), self.target(lemma)

    def input_vector_size(self):
        return self._data.shape[1]

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
    def __init__(self, train_dataset_path, test_dataset_path, dtype=np.float32):
        self.train_dataset = SenseCorpusDataset(train_dataset_path, dtype)
        self.test_dataset = SenseCorpusDataset(test_dataset_path, dtype)
