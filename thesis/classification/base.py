# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import numpy as np


class BaselineClassifier(object):
    def __init__(self):
        self._most_frequent_class = None
        self._most_frequent_class_index = 0
        self._num_classes = 0

    def fit(self, x, y):
        classes, count = np.unique(y, return_counts=True)

        self._most_frequent_class_index = np.argmax(count)
        self._most_frequent_class = classes[self._most_frequent_class_index]
        self._num_classes = classes.shape[0]

        return self

    def predict(self, x):
        return np.array([self._most_frequent_class] * x.shape[0])

    def predict_proba(self, x):
        probability_array = np.zeros((x.shape[0], self._num_classes), dtype=np.float32)

        probability_array[:, self._most_frequent_class_index] = 1.0

        return probability_array
