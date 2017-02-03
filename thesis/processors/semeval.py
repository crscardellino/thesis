# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import itertools
import nltk
import sys
from thesis.utils import find
from lxml import etree
from nltk.stem import WordNetLemmatizer
from thesis.processors.base import XMLCorpusReader

if sys.version_info.major == 2:  # For Python 2 use lazy map
    from itertools import imap as map
else:  # For Python 3 unicode == str
    unicode = str

_lemmatizer = WordNetLemmatizer()


class SemevalTrainCorpusReader(XMLCorpusReader):
    """
    Reader for the Semeval 2007 Task 17 Subtask 1 Training Corpus.
    Intended to return only some of the information available,
    namely: word, lemma, pos.
    """

    _default_files = 'english-lexical-sample.train.xml'

    def _sents(self):
        for doc in self._parse_docs():
            yield doc.xpath('.//instance')

    @property
    def sentences(self):
        return map(SemevalTrainCorpusReader.parsed, itertools.chain(*self._sents()))

    @property
    def words(self):
        return (tuple([unicode(idx)] + word[1:])
                for idx, word in enumerate(itertools.chain(*self.sentences), start=1))

    @staticmethod
    def parsed(sentence):
        def lemmatize(word, pos):
            word_lemma = word.lower()

            if pos.startswith('JJ'):
                word_lemma = _lemmatizer.lemmatize(word_lemma, 'a')
            elif pos.startswith('RG'):
                word_lemma = _lemmatizer.lemmatize(word_lemma, 'r')
            elif pos.startswith('NN'):
                word_lemma = _lemmatizer.lemmatize(word_lemma, 'n')
            elif pos.startswith('VB'):
                word_lemma = _lemmatizer.lemmatize(word_lemma, 'v')

            return word_lemma

        main_lemma, lemma_tag = sentence.getparent().get('item', '-.-').split('.')
        main_lemma_index = len(sentence.find('context').text.strip().split()) + 1

        metadata = dict(
            main_lemma=main_lemma,
            lemma_tag=lemma_tag,
            main_lemma_index='%d' % main_lemma_index,  # FIXME: A better way to do this?
            sense=sentence.find('answer').get('senseid', '-') if sentence.find('answer') is not None else '-',
            resource_sentence=sentence.get('id', '-'),
            doc=sentence.get('docsrc', '-'),
        )

        etree.strip_tags(sentence.find('context'), 'head')
        tagged_context = nltk.pos_tag(sentence.find('context').text.strip().split())

        words = [
            (unicode(idx),
             word,
             lemmatize(word, pos),
             pos)
            for idx, (word, pos) in enumerate(tagged_context, start=1)
            ]

        return sorted(metadata.items()), words

    def __repr__(self):
        return '<SemevalCorpusReader>'


class SemevalTestCorpusReader(SemevalTrainCorpusReader):
    """
    Reader for the Semeval 2007 Task 17 Subtask 1 Test Corpus.
    Intended to return only some of the information available,
    namely: word, lemma, pos.
    """

    _default_files = 'english-lexical-sample.test.xml'
    _key_file = 'english-lexical-sample.test.key'

    def __init__(self, path, files=None):
        super(SemevalTestCorpusReader, self).__init__(path, files)

        self._test_results = {}
        with open(next(find(self._path, self._key_file)), 'r') as f:
            for line in f:
                main_lemma, sentenceid, sense = line.strip().split()
                main_lemma, lemma_tag = main_lemma.split('.')

                self._test_results[(main_lemma, lemma_tag, sentenceid)] = sense

    @property
    def sentences(self):
        return map(self.test_parsed, itertools.chain(*self._sents()))

    def test_parsed(self, sentence):
        metadata, words = self.parsed(sentence)
        metadata = dict(metadata)

        main_lemma, lemma_tag = sentence.getparent().get('item', '-.-').split('.')
        sentenceid = sentence.get('id')
        metadata['sense'] = self._test_results[(main_lemma, lemma_tag, sentenceid)]

        return sorted(metadata.items()), words
