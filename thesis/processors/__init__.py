# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from thesis.processors.ancora import AncoraCorpusReader
from thesis.processors.semeval import SemevalTestCorpusReader, SemevalTrainCorpusReader
from thesis.processors.sensem import SenSemCorpusReader


__all__ = ['AncoraCorpusReader', 'SemevalTestCorpusReader', 'SemevalTrainCorpusReader', 'SenSemCorpusReader']
