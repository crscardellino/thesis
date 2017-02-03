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
                        type=str,
                        help='SenSem column file path')
    parser.add_argument('--output',
                        type=str,
                        default=None,
                        help='Output file to write (defaults to STDOUT)')
    parser.add_argument('--sentences',
                        type=int,
                        default=24153,
                        help='Number of sentences to parse')
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='Number of sentences to batch analyze.')
    args = parser.parse_args()

    output = sys.stdout if args.output is None else open(args.output, 'w')

    parser = ColumnCorpusParser(args.corpus, 'idx', 'token', 'lemma', 'tag',
                                'short_tag', 'morpho_info', 'nec', 'sense', 'syntax',
                                'dephead', 'deprel', 'coref', 'srl')

    tokenized_freeling_sentences = []
    sentences = []

    pbar = tqdm(total=args.sentences)

    for sidx, sentence in enumerate(parser.sentences, start=1):
        tokenized_sentence = [word.token for word in sentence]
        tokenized_sentence = [token for widx, token in enumerate(tokenized_sentence)
                              if widx == 0 or (widx > 0 and tokenized_sentence[widx-1].lower() != token.lower())]
        tokenized_sentence = '\n'.join(tokenized_sentence)

        tokenized_freeling_sentences.append(tokenized_sentence)
        sentences.append(sentence)

        if (sidx % args.batch_size == 0) or (sidx == args.sentences):
            tokenized_freeling_sentences = '\n\n'.join(tokenized_freeling_sentences) + '\n\n'

            freeling_args = (
                '-f', 'es.cfg',
                '--input', 'freeling',
                '--inplv', 'splitted',
                '--output', 'conll',
                '--outlv', 'dep',
                '--ner', '--nec', '--loc',
                '--dep', 'treeler'
            )

            parsed_sentences = sh.analyze(*freeling_args, _in=tokenized_freeling_sentences)
            parsed_sentences = sh.awk('/^\s*$/ { print } { print $1, $2, $3, $4, $5, $6, $7, $10, $11 }',
                                      _in=parsed_sentences)

            for sentence, parsed_sentence in zip(sentences, parsed_sentences.strip().split('\n\n')):
                parsed_sentence = parsed_sentence.strip().split('\n')
                parsed_sentence_lemma = parsed_sentence[sentence.main_lemma_index-1].strip().split()[2]

                if parsed_sentence_lemma != sentence.main_lemma:
                    tqdm.write('NOT FOUND LEMMA for sentence %s (%s != %s)'
                               % (sentence.sentence_index, sentence.main_lemma, parsed_sentence_lemma))
                    sentence['main_lemma_index'] = '-'

                printing_sentence = '\n'.join(parsed_sentence)
                printing_sentence = sh.column('-t', _in=printing_sentence.strip() + '\n')

                print(sentence.metadata_string, file=output)
                print(printing_sentence.strip(), file=output, end='\n\n')

            tokenized_freeling_sentences = []
            sentences = []
            pbar.update(sidx)

    if args.output is not None:
        output.close()
 
    pbar.close()

    print('SenSem corpus parsed', file=sys.stderr)
