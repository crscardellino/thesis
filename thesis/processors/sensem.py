# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import itertools
import sys
from thesis.processors.base import XMLCorpusReader

if sys.version_info.major == 2:  # For Python 2 use lazy map
    from itertools import imap as map
else:  # For Python 3 unicode == str
    unicode = str


VERB_POSITION_ERROR_MARGIN = 2


def in_between_with_margin(mid, min_, max_):
    mid = int(float(mid))

    return min_ - VERB_POSITION_ERROR_MARGIN <= mid <= max_ + VERB_POSITION_ERROR_MARGIN


class SenSemCorpusReader(XMLCorpusReader):
    """
    Reader for the SenSem Corpus. Intended to return only some of the
    information available, namely: word, lemma, pos, sense.
    """

    _default_files = 'spsemcor.utf8.xml'

    def _sents(self):
        for doc in self._parse_docs():
            yield doc.findall('s')

    @property
    def sentences(self):
        return map(SenSemCorpusReader.parsed, itertools.chain(*self._sents()))

    @property
    def words(self):
        return (tuple([unicode(idx)] + word[1:])
                for idx, word in enumerate(itertools.chain(*(word for _, word in self.sentences)), start=1))

    @staticmethod
    def parsed(sentence):
        verb_lemma = sentence.get('lema_verbo', '-')

        metadata = dict(
            verb_lemma=verb_lemma,
            sensem_sense=sentence.get('sentido', '-'),
            sensem_sentence=sentence.get('id', '-'),
            wn=sentence.get('WN_S', '-'),
            wn16=sentence.get('WN16_S', '-'),
            wn30=sentence.get('WN30_S', '-')
        )

        # Look for the verb in the natural candidates
        # FIXME: This is not nice, but works. Maybe an alternative?
        verb_positions = '|' + '|'.join(sentence.get('verbo', '').split(',')) + '|'

        xpath_attribute_query = "contains('%s', concat('|', @sensem_id, '|')) or " % verb_positions
        xpath_attribute_query += "contains('%s', concat('|', @id, '|'))" % verb_positions

        verb_candidates = [v for v in sentence.xpath('.//w[%s]' % xpath_attribute_query)
                           if v.get('lema') == verb_lemma]

        metadata['verb_position'] = verb_candidates[0].get('id') if len(verb_candidates) > 0 else '-'

        # If the verb is not where it should be we follow different ways to obtain it
        if metadata['verb_position'] is '-':
            verb_candidates = [v for v in sentence.xpath(".//w[@lema='%s']" % verb_lemma)]

            if len(verb_candidates) == 1:
                metadata['verb_position'] = verb_candidates[0].get('id')
            elif len(verb_candidates) == 0 or sentence.get('verbo') is None:
                # FIXME: There is no real information on how to look for the verb. We give up.
                metadata['verb_position'] = '-'
            else:
                verb_positions = [int(v) for v in sentence.get('verbo', '').split(',')]
                verb_candidates = [v for v in verb_candidates
                                   if in_between_with_margin(v.get('id_sensem'),
                                                             verb_positions[0],
                                                             verb_positions[-1])]

                if len(verb_candidates) == 1:
                    metadata['verb_position'] = verb_candidates[0].get('id')
                else:
                    # FIXME: The verb is among a possibility of examples. Take them all.
                    metadata['verb_position'] = ','.join([v.get('id') for v in verb_candidates])

        words = [
            (word.get('id'),
             word.get('forma'),
             word.get('lema', '-'),
             word.get('etiqueta', '-'),
             word.get('WN16_S', '-'),
             word.get('WN30_S', '-'))
            for word in sentence.xpath('.//w')
            if word.get('forma') is not None  # FIXME: skip empty words. Is this the better way?
        ]

        return sorted(metadata.items()), words

    def __repr__(self):
        return '<SenSemCorpusReader>'
