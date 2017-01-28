# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from thesis.utils import find
from lxml import etree


class XMLCorpusReader(object):
    """
    Base class for corpus readers in XML format.
    """

    _default_files = '*'

    def __init__(self, path, files=None):
        self._path = path
        if files is None:
            self._files = self._default_files
        else:
            self._files = files

    @property
    def files(self):
        for file_path in find(self._path, self._files):
            yield file_path

    def _parse_docs(self):
        for filename in self.files:
            yield etree.parse(filename).getroot()

    def _sents(self):
        raise NotImplementedError

    @property
    def sentences(self):
        raise NotImplementedError

    @property
    def words(self):
        raise NotImplementedError

    @staticmethod
    def parsed(sentence):
        """ Extracts the information from the xml element """
        raise NotImplementedError

    def __repr__(self):
        return '<XMLCorpusReader>'

