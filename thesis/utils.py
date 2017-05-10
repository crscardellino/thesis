# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import fnmatch
import numpy as np
import os


SENSEM_COLUMNS = ('idx', 'token', 'lemma', 'tag', 'morpho_info', 'ner', 'dep_head', 'dep_rel')
SEMEVAL_COLUMNS = ('idx', 'token', 'lemma', 'tag', 'ner', 'dep_head', 'dep_rel')

RANDOM_SEED = 1234


def find(path, pattern):
    """
    Implementation of unix `find`
    :param path: Path to traverse
    :param pattern: File pattern to look for
    :return: Generator traversing the path yielding files matching the pattern
    """

    for root, _, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)


def try_number(item):
    try:
        return int(item)
    except ValueError:
        pass
    try:
        return float(item)
    except ValueError:
        return item


def cumulative_index_split(target, splits=3, min_count=2):
    # Ensure there is at least 'min_count' items per class in the first split
    split_begin = np.concatenate([np.where(target == label)[0][:min_count] for label in np.unique(target)])
    mask = np.ones_like(target, dtype=np.bool)
    mask[split_begin] = False
    index_accumulator = []

    # Yield each split appended to the previous ones
    for spi in np.array_split(np.concatenate((split_begin, np.arange(target.shape[0])[mask])), splits):
        index_accumulator.extend(spi)
        yield index_accumulator
