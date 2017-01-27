#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import os
import sys

from corpora.parsers import ColumnCorpusParser, Freeling
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('corpus',
                        help='SenSem column file path')
    parser.add_argument('--output',
                        default=None,
                        help='Output file to write (defaults to STDOUT)')
    parser.add_argument('--sentences',
                        default=24207,
                        help='Number of sentences to parse')
    args = parser.parse_args()

    output = sys.stdout if args.output is None else open(args.output, 'w')

    parser = ColumnCorpusParser(args.corpus, 'idx', 'token', 'lemma', 'pos')

    freeling = Freeling(language='es', input_format='freeling', input_level='tagged', output_format='conll',
                        output_level='dep', multiword=True, ner=True, nec=True)

    freeling_sentences = []
    sentences_metadata = []
    for sidx, sentence in enumerate(tqdm(parser.sentences, total=args.sentences), start=1):
        freeling_sentence = []
        word = ''
        for wrd in sentence:
            word = wrd.token
            word += '\t%s' % wrd.lemma

            if sentence.verb_position == str(wrd.idx):
                word += '|main_verb'

            word += '\t%s' % wrd.pos
            word += '\t1'

            freeling_sentence.append(word)

        freeling_sentences.append(freeling_sentence)
        sentences_metadata.append(sentence.metadata_string)

        if (sidx % 1000 == 0) or (sidx == args.sentences):
            (parsed_sentences, parsed_errors), returncode = freeling.run(freeling_sentences)
            parsed_sentences = parsed_sentences.decode('utf-8')
            parsed_errors = parsed_errors.decode('utf-8')
         
            if returncode != os.EX_OK:
                tqdm.write('Parser error: %s' % parsed_errors, file=sys.stderr)
                sys.exit(returncode)

            for metadata, parsed_sentence in zip(sentences_metadata, parsed_sentences.strip().split('\n\n')):
                print(metadata, file=output)
                print(parsed_sentence.strip(), file=output, end='\n\n')

            freeling_sentences = []
            sentences_metadata = []

    if args.output is not None:
        output.close()

    print('SenSem corpus parsed', file=sys.stderr)
