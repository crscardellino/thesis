# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import numpy as np
import pickle

from collections import namedtuple
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix


_InstanceId = namedtuple('InstanceId', 'corpus file sentence lemma idx')


class CorpusDataset(object):
    def __init__(self, dataset, feature_dict_path=None, word_vector_model=None,
                 dataset_extra=None, dtype=np.float32):
        if feature_dict_path is not None:
            with open(feature_dict_path, 'rb') as f:
                self._features_dicts = pickle.load(f)

        if word_vector_model is None or dataset_extra is not None:
            self._data = csr_matrix((dataset['data'], dataset['indices'], dataset['indptr']),
                                    shape=dataset['shape'], dtype=dtype)
            self._input_vector_size = self._data.shape[1]
            self._data_extra = None
            self._word_vector_model = None

            if dataset_extra is not None:
                self._data_extra = dataset_extra['data']
                self._word_vector_model = word_vector_model
                try:
                    self._word_vector_size = self._word_vector_model.vector_size
                except AttributeError:
                    self._word_vector_size = next(iter(self._word_vector_model.values())).shape[0]

                self._input_vector_size += self._data_extra.shape[1] * self._word_vector_size
        else:
            self._data = dataset['data']
            self._data_extra = None
            self._word_vector_model = word_vector_model

            try:
                self._word_vector_size = self._word_vector_model.vector_size
            except AttributeError:
                self._word_vector_size = next(iter(self._word_vector_model.values())).shape[0]

            self._input_vector_size = self._data.shape[1] * self._word_vector_size

        self._lemmas = None
        self._unique_lemmas = None
        self.dtype = dtype

    def _word_window_to_vector(self, word_window):
        vector = []

        for word in word_window:
            try:
                vector.append(self._word_vector_model[next(t for t in word if t in self._word_vector_model)])
            except StopIteration:
                vector.append(np.zeros(self._word_vector_size, dtype=self.dtype))

        return np.concatenate(vector)

    def data(self, lemma=None, limit=0):
        data = self._data if lemma is None else self._data[np.where(self._lemmas == lemma)[0], :]
        extra_data = None

        if self._word_vector_model is not None:
            if self._data_extra is None:
                data = np.array([self._word_window_to_vector(ww) for ww in data])
            else:
                extra_data = self._data_extra if lemma is None \
                    else self._data_extra[np.where(self._lemmas == lemma)[0], :]
                extra_data = np.array([self._word_window_to_vector(ww) for ww in extra_data])

        if limit > 0:
            data = data[:limit, :]
            if extra_data is not None:
                extra_data = extra_data[:limit, :]
                data = np.hstack((data.toarray(), extra_data))

        return data

    def input_vector_size(self):
        return self._input_vector_size

    def num_examples(self, lemma=None):
        return self.data(lemma).shape[0]

    @property
    def num_lemmas(self):
        return self._unique_lemmas.shape[0]

    @property
    def word_vector_model(self):
        return self._word_vector_model

    def features_dictionaries(self, lemma=None, limit=0):
        if lemma is None:
            features_dict = self._features_dicts
        else:
            instances = set(np.where(self._lemmas == lemma)[0])
            features_dict = [fd for idx, fd in enumerate(self._features_dicts) if idx in instances]

        if limit > 0:
            features_dict = features_dict[:limit]

        return features_dict


class SenseCorpusDataset(CorpusDataset):
    def __init__(self, dataset_path, features_dict_path=None, word_vector_model=None,
                 dataset_extra=None, dtype=np.float32):
        dataset = np.load(dataset_path)
        dataset_extra = np.load(dataset_extra) if dataset_extra is not None else None
        super(SenseCorpusDataset, self).__init__(dataset, features_dict_path, word_vector_model, dataset_extra, dtype)

        self._lemmas = dataset['lemmas']
        self._unique_lemmas = np.unique(self._lemmas)
        self._target = dataset['target']
        self._sentences = dataset['sentences']
        self._train_classes = dataset['train_classes']

    def target(self, lemma=None, limit=0):
        if lemma is None:
            target = self._target
        else:
            target = self._target[np.where(self._lemmas == lemma)[0]]

        if limit > 0:
            target = target[:limit, :]

        return target

    def traverse_dataset_by_lemma(self, return_features=False):
        for lemma in self._unique_lemmas:
            if return_features:
                yield lemma, self.data(lemma), self.target(lemma), self.features_dictionaries(lemma)
            else:
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

    @property
    def train_classes(self):
        return self._train_classes


class SenseCorpusDatasets(object):
    def __init__(self, train_dataset_path, test_dataset_path, train_features_dict_path=None,
                 test_features_dict_path=None, word_vector_model_path=None,
                 train_dataset_extra=None, test_dataset_extra=None, dtype=np.float32):
        try:
            word_vector_model = Word2Vec.load_word2vec_format(word_vector_model_path, binary=True)\
                if word_vector_model_path is not None else None
        except UnicodeDecodeError:
            with open(word_vector_model_path, 'rb') as fvectors:
                word_vector_model = pickle.load(fvectors)

        self.train_dataset = SenseCorpusDataset(train_dataset_path, train_features_dict_path,
                                                word_vector_model, train_dataset_extra, dtype)
        self.test_dataset = SenseCorpusDataset(test_dataset_path, test_features_dict_path,
                                               word_vector_model, test_dataset_extra, dtype)


class UnlabeledCorpusDataset(CorpusDataset):
    def __init__(self, dataset_path, features_dict_path=None, word_vector_model=None,
                 dataset_extra=None, dtype=np.float32):
        dataset = np.load(dataset_path)
        super(UnlabeledCorpusDataset, self).__init__(dataset, features_dict_path, word_vector_model,
                                                     dataset_extra, dtype)

        self._instances_id = [_InstanceId(*iid.split(':')) for iid in dataset['instances_id']]
        self._lemmas = np.array([iid.lemma for iid in self._instances_id])
        self._unique_lemmas = np.unique(self._lemmas)

    def instances_id(self, lemma=None, limit=0):
        if lemma is None:
            instances_id = self._instances_id
        else:
            instances = set(np.where(self._lemmas == lemma)[0])
            instances_id = [iid for idx, iid in enumerate(self._instances_id) if idx in instances]

        if limit > 0:
            instances_id = instances_id[:limit]

        return instances_id

    def has_lemma(self, lemma):
        return lemma in set(self._unique_lemmas)
