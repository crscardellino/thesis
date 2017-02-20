# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import numpy as np


class BaselineClassifier(object):
    def __init__(self):
        self._most_frequent_class = None

    def fit(self, x, y):
        classes, count = np.unique(y, return_counts=True)

        self._most_frequent_class = classes[np.argmax(count)]

        return self

    def predict(self, x):
        return np.array([self._most_frequent_class] * x.shape[0])
