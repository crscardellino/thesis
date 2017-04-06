# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from thesis.utils import RANDOM_SEED


def filter_minimum(data, target, min_count=2, invalid_target=-1):
    valid_targets = np.where(target != invalid_target)[0]

    data = data[valid_targets]
    target = target[valid_targets]

    labels, counts = np.unique(target, return_counts=True)
    over_minimum_count = np.where(counts >= min_count)

    if over_minimum_count.shape[0] < 2:
        raise ValueError('Not enough labels to cover minimum count')

    filtered_by_count = np.in1d(target, labels[over_minimum_count])

    return data[filtered_by_count], target[filtered_by_count]


def validation_split(data, target, validation_ratio=0.1, force_split=True, random_seed=RANDOM_SEED):
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=random_seed)
        train_index, validation_index = next(sss.split(data, target))
    except ValueError:
        if not force_split:
            raise
        num_classes = np.unique(target).shape[0]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=num_classes / np.float(target.shape[0]),
                                     random_state=random_seed)
        train_index, validation_index = next(sss.split(data, target))

    return data[train_index], data[validation_index], target[train_index], target[validation_index]
