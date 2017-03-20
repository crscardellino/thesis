# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import argparse
import numpy as np
import os
import pickle
import scipy.sparse as sps
import sh
import sys
import unicodedata

from collections import defaultdict
from itertools import chain
from thesis.parsers import Sentence
from tqdm import tqdm


class WordCoocurrenceMatrixEmbeddings(object):
    def __init__(self, k=100, window_size=5, lowercase=False, min_count=5):
        self._K = k
        self._window_size = window_size
        self._lowercase = lowercase
        self._min_count = min_count
        self._word_cooccurrences = defaultdict(lambda: defaultdict(int))
        self._word_counter = defaultdict(int)

    def _fit(self, sentences):
        """
        Creates the word_cooccurrences dictionary from sentences.
        :type sentences: list(thesis.parsers.Sentence)
        :return: instance of self
        """

        for sentence in sentences:
            for main_word in sentence:
                main_word_token = main_word.token.lower() if self._lowercase else main_word.token
                self._word_counter[main_word_token] += 1
                for window_word in chain(*sentence.get_word_windows(main_word.idx, self._window_size)):
                    window_word_token = window_word.token.lower() if self._lowercase else window_word.token
                    self._word_cooccurrences[main_word_token][window_word_token] += 1

        return self

    def _transform(self):
        """
        Function to return the embeddings (as a dictionary) using CSV with K values.
        :return: matrix of word coocurrences in scipy.sparse.csr_matrix format
        """

        if self._min_count > 0:  # Filter cases of low occurrence
            self._word_cooccurrences = {word: coocurrences
                                        for word, coocurrences in self._word_cooccurrences.items()
                                        if self._word_counter[word] > self._min_count}

        vocabulary = {word: widx for widx, word in enumerate(sorted(self._word_cooccurrences))}
        voc_size = len(vocabulary)
        rows, cols, data = [], [], []

        for main_word, coocurrences in self._word_cooccurrences.items():
            for window_word, value in coocurrences.items():
                if self._min_count == 0 or self._word_counter[window_word] > self._min_count:
                    rows.append(vocabulary[main_word])
                    cols.append(vocabulary[window_word])
                    data.append(value)

        word_window_matrix = sps.coo_matrix((data, (rows, cols)),
                                            shape=(voc_size, voc_size),
                                            dtype=np.float32).tocsr()

        word_embeddings_matrix = sps.linalg.svds(word_window_matrix, k=self._K)[0]
        return {word: word_embeddings_matrix[vocabulary[word], :] for word in vocabulary}

    def fit_transform(self, sentences):
        self._fit(sentences)
        return self._transform()


def _traverse_corpus(path):
    for fname in sh.find(path, '-type', 'f'):
        with open(fname.strip(), 'r') as fin:
            corpus_name = '%s.%s' % (os.path.basename(os.path.dirname(fname)),
                                     os.path.basename(fname))
            sentence_no = 1
            word_no = 1
            sentence = []
            metadata = {'META': corpus_name, 'sentence': sentence_no}
            for line in tqdm(fin):
                if line.strip() == '':
                    yield Sentence(metadata, sentence, 'idx', 'token', 'lemma', 'tag')

                    sentence_no += 1
                    word_no = 1
                    sentence = []
                    metadata = {'META': corpus_name, 'sentence': sentence_no}
                else:
                    try:
                        token, lemma, tag, _ = unicodedata.normalize('NFC', line).strip().split()

                        if tag.startswith('F'):
                            token = '<PUNCTUATION>'
                        elif tag.startswith('W'):
                            token = '<DATE>'
                        elif tag.startswith('Z'):
                            token = '<NUMBER>'

                        sentence.append('%d\t%s\t%s\t%s' % (word_no, token, lemma, tag))
                        word_no += 1
                    except ValueError:
                        continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('path',
                        type=str,
                        help='Path to root directory of corpus to parse.')
    parser.add_argument('save_path',
                        type=str,
                        help='Save path of the pickle file (dictionary with vectors).')
    parser.add_argument('--size',
                        type=int,
                        default=100,
                        help='Embeddings dimensions.')
    parser.add_argument('--window_size',
                        type=int,
                        default=5,
                        help='Window size.')
    parser.add_argument('--lowercase',
                        action='store_true',
                        help='Whether to lowercase the words or not.')
    parser.add_argument('--min_count',
                        type=int,
                        default=3,
                        help='Minimum number of occurrences to consider a word.')

    args = parser.parse_args()

    word_coocurrence_model = WordCoocurrenceMatrixEmbeddings(k=args.size, window_size=args.window_size,
                                                             lowercase=args.lowercase, min_count=args.min_count)

    print('Fitting dataset', file=sys.stderr)
    word_embeddings = word_coocurrence_model.fit_transform(_traverse_corpus(args.path))
    print('All dataset fit', file=sys.stderr)

    print('Saving embeddings to pickle file %s' % args.save_path, file=sys.stderr)
    with open(args.save_path, 'wb') as fout:
        pickle.dump(word_embeddings, fout)
