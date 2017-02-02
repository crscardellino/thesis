#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import sh
import sys

from thesis.parsers import ColumnCorpusParser
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('corpus',
                        help='SenSem column file path')
    parser.add_argument('--output',
                        default=None,
                        help='Output file to write (defaults to STDOUT)')
    parser.add_argument('--sentences',
                        default=24153,
                        help='Number of sentences to parse')
    args = parser.parse_args()

    output = sys.stdout if args.output is None else open(args.output, 'w')

    parser = ColumnCorpusParser(args.corpus, 'idx', 'token', 'lemma', 'pos',
                                'short_pos', 'morpho_info', 'nec', 'sense', 'syntax',
                                'dephead', 'deprel', 'coref', 'srl')

    for sidx, sentence in enumerate(tqdm(parser.sentences, total=args.sentences), start=1):
        tokenized_sentence = [word.token for word in sentence]
        tokenized_sentence = [token for widx, token in enumerate(sentence)
                              if widx == 0 or (widx > 0 and sentence[widx-1].lower() != token.lower())]
        tokenized_sentence = '\n'.join(tokenized_sentence)

        freeling_args = (
            '-f', 'es.cfg',
            '--input', 'freeling',
            '--inplv', 'splitted',
            '--output', 'conll',
            '--outlv', 'dep',
            '--ner', '--nec', '--loc',
            '--dep', 'treeler'
        )

        parsed_sentence = sh.analyze(*freeling_args, _in=tokenized_sentence)
        parsed_sentence = sh.awk('{ print $1, $2, $3, $4, $5, $6, $7, $10, $11 }', _in=parsed_sentence)

        parsed_sentence = parsed_sentence.strip().split('\n')
        parsed_sentence_lemma = parsed_sentence[sentence.verb_position].strip().split()[2]

        if parsed_sentence_lemma != sentence.lemma:
            tqdm.write('NOT FOUND LEMMA for sentence %s (%s != %s)'
                       % (sentence.sentence_index, sentence.lemma, parsed_sentence_lemma), file=sys.stdout)
            sentence['verb_position'] = '-'

        printing_sentence = '\n'.join(parsed_sentence)
        printing_sentence = sh.column('-t', _in=printing_sentence.strip() + '\n')

        print(sentence.metadata_string, file=output)
        print(printing_sentence.strip(), file=output, end='\n\n')

    if args.output is not None:
        output.close()

    print('SenSem corpus parsed', file=sys.stderr)
