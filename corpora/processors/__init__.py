# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from corpora.processors.ancora import AncoraCorpusReader
from corpora.processors.semeval import SemevalTestCorpusReader, SemevalTrainCorpusReader
from corpora.processors.sensem import SenSemCorpusReader


__all__ = ['AncoraCorpusReader', 'SemevalTestCorpusReader', 'SemevalTrainCorpusReader', 'SenSemCorpusReader']
