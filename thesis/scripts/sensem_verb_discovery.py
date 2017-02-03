#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import sys

from thesis.parsers import ColumnCorpusParser
from tqdm import tqdm


def search_main_verb(sentence):
    if sentence.main_lemma_index.isdigit() and sentence.main_lemma_index < len(sentence):
        word = sentence.get_word_by_index(sentence.main_lemma_index)
        lemma = word.lemma.split('|')

        if lemma[-1] == 'main_verb':
            word.lemma = lemma[0]
            return 0

    for word in sentence:
        lemma = word.lemma.split('|')

        if lemma[-1] == 'main_verb':
            word.lemma = lemma[0]
            return word.idx

    return -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('corpus',
                        help='SenSem column file path')
    parser.add_argument('--output',
                        default=None,
                        help='Output file to write (defaults to STDOUT)')
    parser.add_argument('--logfile',
                        default='discovery.log',
                        help='File to write the information about sentences.')
    parser.add_argument('--sentences',
                        default=24207,
                        help='Number of sentences to parse')
    args = parser.parse_args()

    output = sys.stdout if args.output is None else open(args.output, 'w')

    parser = ColumnCorpusParser(args.corpus, 'idx', 'token', 'lemma', 'pos', 'short_pos', 'extended_pos',
                                'ne', 'sense', 'parse', 'head', 'dep', 'extra_a', 'extra_b')

    with open(args.logfile, 'w') as flog:
        for sidx, sentence in enumerate(tqdm(parser.sentences, total=args.sentences), start=1):
            main_verb_index = search_main_verb(sentence)

            if main_verb_index == 0:
                print('NO_CHANGE: Main verb position hasn\'t change for sentence %s' % sentence.sentence_index,
                      file=flog)
            elif main_verb_index > 0:
                print('CHANGE: Main verb changed from position %s to position %d in sentence %s' %
                      (sentence.main_lemma_index, main_verb_index, sentence.sentence_index), file=flog)
                sentence['main_lemma_index'] = str(main_verb_index)
            else:
                print('NOT_FOUND: Main verb position wasn\'t found for sentence %s' % sentence.sentence_index,
                      file=flog)

            print(sentence.metadata_string, file=output)
            print(str(sentence), file=output)
            print('', file=output)

    if args.output is not None:
        output.close()

    print('Finished processing', file=sys.stderr)
