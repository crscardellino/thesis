# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from thesis.constants import RANDOM_SEED


class NotEnoughSensesError(ValueError):
    pass


def filter_minimum(target, min_count=2, invalid_target=-1):
    valid_targets = np.where(target != invalid_target)[0]

    labels, counts = np.unique(target[valid_targets], return_counts=True)
    over_minimum_count = np.where(counts >= min_count)[0]

    if over_minimum_count.shape[0] < 2:
        raise NotEnoughSensesError('Not enough labels to cover minimum count')

    return np.arange(target.shape[0])[np.in1d(target, labels[over_minimum_count])]


def validation_split(target, validation_ratio=0.1, force_split=True, random_seed=RANDOM_SEED):
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=random_seed)
        return next(sss.split(np.zeros_like(target), target))
    except ValueError:
        if not force_split:
            raise
        num_classes = np.unique(target).shape[0]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=num_classes / np.float(target.shape[0]),
                                     random_state=random_seed)
        return next(sss.split(np.zeros_like(target), target))
