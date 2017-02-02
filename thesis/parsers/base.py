# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from collections import OrderedDict


def _try_number(item):
    try:
        return int(item)
    except ValueError:
        pass
    try:
        return float(item)
    except ValueError:
        return item


class Word(object):
    def __init__(self, word, columns):
        word = OrderedDict(zip(columns, word.split()))
        self.idx = int(word.pop('idx'))
        self.token = word.pop('token')
        self.lemma = word.pop('lemma')
        self.tag = word.pop('tag')

        self._extras = word.copy()

    def __contains__(self, item):
        return item in self._extras

    def __getitem__(self, item):
        return _try_number(self._extras[item])

    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            raise AttributeError

    def __setitem__(self, key, value):
        self._extras[key] = value

    def __repr__(self):
        return '<Word: %s>' % self.token

    def __str__(self):
        return '%s\t%s\t%s\t%s\t%s' % (self.idx, self.token, self.lemma, self.tag, '\t'.join(self._extras.values()))


class Sentence(object):
    def __init__(self, metadata, sentence, *columns):
        self._corpus_name = metadata.pop('META')
        self._sentence_index = int(metadata.pop('sentence'))
        self._metadata = metadata
        self._words = [Word(word, columns) for word in sentence]

    def __contains__(self, item):
        return item in self._metadata

    def __getitem__(self, item):
        return _try_number(self._metadata[item])

    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            raise AttributeError

    def __len__(self):
        return len(self._words)

    def __setitem__(self, key, value):
        self._metadata[key] = value

    def __iter__(self):
        for word in self._words:
            yield word

    def __repr__(self):
        return '<Sentence: %05d - Corpus: %s>' % (self._sentence_index, self._corpus_name)

    def __str__(self):
        return '\n'.join(str(word) for word in self)

    @property
    def metadata_string(self):
        return 'META:%s\tsentence:%05d\t%s' % \
               (self._corpus_name, self._sentence_index,
                '\t'.join(':'.join(d) for d in sorted(self._metadata.items())))

    def get_word_by_index(self, index):
        index = int(index)
        assert index > 0 and index == self._words[index-1].idx
        return self._words[index-1]

    @property
    def corpus_name(self):
        return self._corpus_name

    @property
    def sentence_index(self):
        return self._sentence_index


class ColumnCorpusParser(object):
    def __init__(self, path, *columns):
        self._path = path
        self._columns = columns

    @property
    def sentences(self):
        with open(self._path, 'r') as f:
            metadata = {}
            sentence = []
            for line in f:
                if line.strip().startswith('META'):
                    metadata = dict([tuple(mdata.split(':', 1)) for mdata in line.strip().split()])
                elif line.strip() != '':
                    sentence.append(line.strip())
                else:
                    yield Sentence(metadata, sentence, *self._columns)
                    metadata = {}
                    sentence = []

    def __repr__(self):
        return '<ColumnCorpusParser>'