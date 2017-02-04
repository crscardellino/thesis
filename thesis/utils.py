# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import fnmatch
import os


SENSEM_COLUMNS = ('idx', 'token', 'lemma', 'tag', 'morpho_info', 'ner', 'dep_head', 'dep_rel')
SEMEVAL_COLUMNS = ('idx', 'token', 'lemma', 'tag', 'ner', 'dep_head', 'dep_rel')


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