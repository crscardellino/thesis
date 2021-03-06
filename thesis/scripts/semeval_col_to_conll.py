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
                        help='Semeval column file path')
    parser.add_argument('--output',
                        default=None,
                        help='Output file to write (defaults to STDOUT)')
    parser.add_argument('--sentences',
                        default=27132,
                        type=int,
                        help='Number of sentences to parse')
    parser.add_argument('--server',
                        default='http://localhost:9000',
                        help='Full http address and port of the server')
    args = parser.parse_args()

    output = sys.stdout if args.output is None else open(args.output, 'w')

    parser = ColumnCorpusParser(args.corpus, 'idx', 'token', 'lemma', 'tag')

    for sidx, sentence in enumerate(tqdm(parser.sentences, total=args.sentences), start=1):
        tokenized_sentences = ' '.join(word.token for word in sentence)

        parsed_sentences = sh.wget('--post-data', '%s' % tokenized_sentences, args.server, '-O', '-')
        parsed_sentences = [ps.split('\n') for ps in parsed_sentences.strip().split('\n\n')]

        original_lemma_idx = sentence.main_lemma_index

        # FIXME: This is savage!!!
        ps_len = [len(ps) for ps in parsed_sentences]
        ps_len = [sum(ps_len[:i]) for i in range(1, len(ps_len) + 1)]
        lemma_sentence_idx = next(sidx for sidx, slen in enumerate(ps_len) if slen >= original_lemma_idx)
        lemma_sentence = parsed_sentences[lemma_sentence_idx]

        if lemma_sentence_idx > 0:
            main_lemma_index = original_lemma_idx - ps_len[lemma_sentence_idx-1] - 1
        else:
            main_lemma_index = original_lemma_idx - 1

        new_token, new_lemma = parsed_sentences[lemma_sentence_idx][main_lemma_index].strip().split()[1:3]

        if new_lemma != sentence.main_lemma and new_token != sentence.main_token:
            tqdm.write('NOT FOUND LEMMA for sentence %s' % sentence.sentence_index, file=sys.stdout)
            printing_sentence = '\n'.join('\n'.join(ps) for ps in parsed_sentences)
        elif new_lemma != sentence.main_lemma and new_token == sentence.main_token:
            tqdm.write('TOKEN FOUND WITH DIFFERENT LEMMA for sentence %s' % sentence.sentence_index, file=sys.stdout)
            printing_sentence = '\n'.join('\n'.join(ps) for ps in parsed_sentences)
        else:
            sentence['main_lemma_index'] = str(main_lemma_index + 1)
            printing_sentence = '\n'.join(parsed_sentences[lemma_sentence_idx])

        printing_sentence = sh.column('-t', _in=printing_sentence.strip() + '\n')

        print(sentence.metadata_string, file=output)
        print(printing_sentence.strip(), file=output, end='\n\n')

    if args.output is not None:
        output.close()

    print('Semeval corpus parsed', file=sys.stderr)
