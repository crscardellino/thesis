# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import itertools
import sys
from thesis.processors.base import XMLCorpusReader

if sys.version_info.major == 2:  # For Python 2 use lazy map
    from itertools import imap as map
else:  # For Python 3 unicode == str
    unicode = str


class AncoraCorpusReader(XMLCorpusReader):
    """
    Reader for the Ancora Corpus. Intended to return only some of the
    information available, namely: word, lemma, pos, sense and ne.

    Based on the code by Franco Luque: 
    https://github.com/PLN-FaMAF/PLN-2015/blob/master/corpus/ancora.py
    """

    _default_files = '*.tbf.xml'

    def _sents(self):
        for doc in self._parse_docs():
            yield doc.findall('sentence')

    @property
    def sentences(self):
        return map(AncoraCorpusReader.parsed, itertools.chain(*self._sents()))

    @property
    def words(self):
        return (tuple([unicode(idx)] + word[1:])
                for idx, word in enumerate(itertools.chain(*self.sentences), start=1))

    @staticmethod
    def parsed(sentence):
        """ Extracts the information from the xml element """
        def parse_pos(pos):
            return pos.capitalize() if pos.startswith('f') or pos.startswith('z') else pos.upper()

        return [
            (unicode(idx),
             word.get('wd'),
             word.get('lem', '-'),
             parse_pos(word.get('pos', '-')),
             word.get('sense', '-'),
             word.get('ne', '-').upper())
            for idx, word in enumerate(sentence.xpath('.//*[@wd]'), start=1)
        ]

    def __repr__(self):
        return '<AncoraCorpusReader>'
