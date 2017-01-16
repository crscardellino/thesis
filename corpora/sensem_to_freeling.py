#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import os
import sys

from corpora.parsers import ColumnCorpusParser
from corpora.parsers.freeling import Freeling


def search_main_verb(sentence):
    for line in sentence.split():
        lemma = line[2].split('|')

        if lemma[-1] == 'main_verb':
            return int(line[0])

    return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("corpus",
                        help="SenSem column file path")
    parser.add_argument("--output",
                        default=None,
                        help="Output file to write (defaults to STDOUT)")
    args = parser.parse_args()

    output = sys.stdout if args.output is None else open(args.output, "w")

    parser = ColumnCorpusParser(args.corpus, 'idx', 'token', 'lemma', 'pos')

    freeling = Freeling(language='es', input_format='freeling', input_level='splitted', output_format='conll',
                        output_level='dep', multiword=True, ner=True, nec=True)

    for sentence in parser.sentences:
        freeling_sentence = []
        word = ''
        for word in sentence:
            word = word.token
            word += '\t%s' % word.lemma

            if sentence.verb_position == word.idx:
                word += '|main_verb'

            word += '\t%s' % word.pos
            word += '\t1'

        freeling_sentence.append(word)

        (parsed_sentence, parsed_errors), returncode = freeling.run(freeling_sentence)

        if returncode != os.EX_OK:
            print('Parser error: %s' % parsed_errors, file=sys.stderr)
            sys.exit(returncode)
        elif parsed_errors.strip() != '':
            print('Parser warnings: %s' % parsed_errors, file=sys.stderr)

        new_index_for_verb = search_main_verb(parsed_sentence)
        if search_main_verb(parsed_sentence) != -1:
            sentence['verb_position'] = str(new_index_for_verb)
        else:
            print('Verb not found. Please select verb index:\n%s' % parsed_sentence, file=sys.stderr)
            sentence['verb_position'] = input('Verb index: ')

        print(sentence.metadata_string, file=output)
        print(parsed_sentence.rstrip(), file=output)
        print('', file=output)

    if args.output is not None:
        output.close()

    print("SenSem corpus parsed", file=sys.stderr)
