# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import numpy as np
import os
import pickle
import sh
import shutil
import sys

from scipy.sparse import vstack
from thesis.feature_extraction import (HandcraftedHashedFeaturesExtractor, WordWindowExtractor)
from thesis.parsers import ColumnCorpusParser
from thesis.constants import DEFAULT_FEATURES, LANGUAGE, CORPUS_COLUMNS
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('corpus',
                        type=str,
                        help='Path to the corpus directory to featurize.')
    parser.add_argument('save_path',
                        type=str,
                        help='Path to directory to save the matrices. Will be overwritten if exists.')
    parser.add_argument('corpus_labels',
                        type=str,
                        help='Path to pickle file to get lemmas from corpora.')
    parser.add_argument('--ww_path',
                        type=str,
                        default=None,
                        help='If given, stores the word windows in the given path.')
    parser.add_argument('--corpus_language',
                        type=str,
                        default='spanish',
                        help='Corpus language name to featurize (spanish/english).')
    parser.add_argument('--max_instances',
                        type=int,
                        default=1000,
                        help='Maximum number of instances per lemma (default: 1000).')
    parser.add_argument('--hashed_features',
                        type=int,
                        default=2**10,
                        help='In case of using hashing trick, number of features after hashing (default: 1024).')
    parser.add_argument('--window_size',
                        type=int,
                        default=5,
                        help='Size of the symmetric word window (default: 5).')
    parser.add_argument('--ignored_features',
                        type=str,
                        nargs='+',
                        default=[],
                        help='Features to ignore.')

    args = parser.parse_args()

    args.ignored_features = [args.ignored_features] \
        if isinstance(args.ignored_features, str) else args.ignored_features

    features = DEFAULT_FEATURES.copy()

    for ignored_feature in args.ignored_features:
        features.pop(ignored_feature, None)

    features['window_size'] = args.window_size

    extractor = HandcraftedHashedFeaturesExtractor(
        n_features=args.hashed_features, return_features_dict=True,
        **features)
    ww_extractor = None

    if args.ww_path is not None:
        ww_extractor = WordWindowExtractor(args.window_size)

    if os.path.isfile(args.save_path):
        os.remove(args.save_path)
    elif os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)

    os.makedirs(args.save_path)

    instances = []
    features = []
    ww_instances = []
    instances_id = []

    with open(args.corpus_labels, 'rb') as f:
        valid_lemmas = pickle.load(f)
        valid_lemmas = {lemma for lemma, (_, count) in
                        valid_lemmas[LANGUAGE[args.corpus_language]]['lemmas'].items()
                        if count[count >= 2].shape[0] > 1}

    print('Getting instances', file=sys.stderr)

    corpora_files = sorted(sh.find(args.corpus, '-type', 'f').strip().split('\n'))

    max_lemma_per_corpus = int(args.max_instances / len(corpora_files)) + 1

    for corpus_file in tqdm(corpora_files):
        corpus_file = corpus_file.strip()
        parser = ColumnCorpusParser(corpus_file, *CORPUS_COLUMNS[LANGUAGE[args.corpus_language]])
        lemmas_for_corpus = {lemma: 0 for lemma in valid_lemmas}

        for sentence in tqdm(parser.sentences):
            for word in (word for word in sentence if word.tag.startswith('VM') and word.lemma in valid_lemmas
                         and lemmas_for_corpus[word.lemma] < max_lemma_per_corpus):
                lemmas_for_corpus[word.lemma] += 1
                sentence['main_lemma'] = word.lemma
                sentence['main_lemma_index'] = word.idx

                instance_features, instance = extractor.instantiate_sentence(sentence)
                instances.append(instance)
                features.append(instance_features)

                if args.ww_path is not None:
                    ww_instances.append(ww_extractor.instantiate_sentence(sentence))

                instances_id.append('%s:%s:%d:%s:%d' %
                                    (sentence.corpus_name, sentence.file, sentence.sentence_index,
                                     sentence.main_lemma, sentence.main_lemma_index))

    print('Saving resources in directory %s' % args.save_path, file=sys.stderr)

    matrix = vstack(instances)
    np.savez_compressed(os.path.join(args.save_path, 'dataset.npz'),
                        data=matrix.data, indices=matrix.indices, indptr=matrix.indptr,
                        shape=matrix.shape, instances_id=np.array(instances_id))
    with open(os.path.join(args.save_path, 'features.p'), 'wb') as f:
        pickle.dump(features, f)

    if args.ww_path is not None:
        matrix = np.array(ww_instances)
        np.savez_compressed(os.path.join(args.ww_path, 'dataset.npz'),
                            data=matrix, instances_id=np.array(instances_id))

    print('Finished', file=sys.stderr)
