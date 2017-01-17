# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals


class Word(object):
    def __init__(self, word, columns):
        word = dict(zip(columns, word.split()))
        self.idx = word.pop('idx')
        self.token = word.pop('token')
        self.lemma = word.pop('lemma')
        self.pos = word.pop('pos')

        self._extras = word.copy()

    def __contains__(self, item):
        return item in self._extras

    def __getitem__(self, item):
        return self._extras[item]

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
        return '%d\t%s\t%s\t%s\t%s' % (self.idx, self.token, self.lemma, self.pos, '\t'.join(self._extras))


class Sentence(object):
    def __init__(self, metadata, sentence, *columns):
        self._corpus_name = metadata.pop('META')
        self._sentence = metadata.pop('sentence')
        self._metadata = metadata
        self._words = [Word(word, columns) for word in sentence]

    def __contains__(self, item):
        return item in self._metadata

    def __getitem__(self, item):
        return self._metadata[item]

    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            raise AttributeError

    def __setitem__(self, key, value):
        self._metadata[key] = value

    def __iter__(self):
        for word in self._words:
            yield word

    def __repr__(self):
        return '<Sentence: %s>' % self._corpus_name

    def __str__(self):
        return '\n'.join(str(word) for word in self)

    @property
    def metadata_string(self):
        return 'META:%s\tsentence:%s\t%s' % \
               (self._corpus_name, self._sentence, '\t'.join(':'.join(d) for d in self._metadata.items()))

    def get_word_by_index(self, index):
        assert index > 0 and index == self._words[index-1].idx
        return self._words[index-1]

    @property
    def corpus_name(self):
        return self._corpus_name


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

    def __repr__(self):
        return '<ColumnCorpusParser>'
