#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import sh
import sys

from corpora.parsers import ColumnCorpusParser
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('corpus',
                        help='SenSem column file path')
    parser.add_argument('--output',
                        default=None,
                        help='Output file to write (defaults to STDOUT)')
    parser.add_argument('--sentences',
                        default=4851,
                        help='Number of sentences to parse')
    parser.add_argument('--server',
                        default='http://localhost:9000',
                        help='Full http address and port of the server')
    args = parser.parse_args()

    output = sys.stdout if args.output is None else open(args.output, 'w')

    parser = ColumnCorpusParser(args.corpus, 'idx', 'token', 'lemma', 'pos')

    for sidx, sentence in enumerate(tqdm(parser.sentences, total=args.sentences), start=1):
        tokenized_sentences = ' '.join(word.token for word in sentence)

        parsed_sentences = sh.curl('--data', '%s' % tokenized_sentences, args.server, '-o', '-')
        parsed_sentences = [ps.split('\n') for ps in parsed_sentences.strip().split('\n\n')]

        original_lemma_idx = int(sentence.lemma_idx)

        # FIXME: This is savage!!!
        ps_len = [len(ps) for ps in parsed_sentences]
        ps_len = [sum(ps_len[:i]) for i in range(1, len(ps_len) + 1)]
        lemma_sentence_idx = next(sidx for sidx, slen in enumerate(ps_len) if slen >= original_lemma_idx)
        lemma_sentence = parsed_sentences[lemma_sentence_idx]

        if lemma_sentence_idx > 0:
            lemma_idx = original_lemma_idx - ps_len[lemma_sentence_idx-1] - 1
        else:
            lemma_idx = original_lemma_idx - 1

        if parsed_sentences[lemma_sentence_idx][lemma_idx].strip().split()[2] != sentence.lemma:
            tqdm.write('NOT FOUND LEMMA for sentence %s' % sentence.sentence_index, file=sys.stdout)
            printing_sentence = '\n'.join('\n'.join(ps) for ps in parsed_sentences)
        else:
            sentence['lemma_idx'] = str(lemma_idx)
            printing_sentence = '\n'.join(parsed_sentences[lemma_sentence_idx])

        printing_sentence = sh.column('-t', _in=printing_sentence.strip() + '\n')

        print(sentence.metadata_string, file=output)
        print(printing_sentence, file=output, end='\n\n')

    if args.output is not None:
        output.close()

    print('SenSem corpus parsed', file=sys.stderr)
