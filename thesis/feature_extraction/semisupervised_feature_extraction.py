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
from thesis.utils import SEMEVAL_COLUMNS, SENSEM_COLUMNS
from tqdm import tqdm


_DEFAULT_FEATURES = {
    'main_token': True, 'main_lemma': True, 'main_tag': True, 'main_morpho': True,
    'window_bow': True, 'window_tokens': True, 'window_lemmas': True, 'window_tags': True,
    'surrounding_bigrams': True, 'surrounding_trigrams': True,
    'inbound_dep_triples': True, 'outbound_dep_triples': True
}

_CORPUS_COLUMNS = {
    'semeval': SEMEVAL_COLUMNS,
    'sensem': SENSEM_COLUMNS
}

_LANGUAGE = {
    'spanish': 'sensem',
    'english': 'semeval'
}


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
    parser.add_argument('--corpus_language',
                        type=str,
                        default='spanish',
                        help='Corpus language name to featurize (spanish/english).')
    parser.add_argument('--max_instances',
                        type=int,
                        default=1000,
                        help='Maximum number of instances per lemma (default: 1000).')
    parser.add_argument('--windowizer',
                        action='store_true',
                        help='If active use word window extractor.')
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

    features = _DEFAULT_FEATURES.copy()

    for ignored_feature in args.ignored_features:
        features.pop(ignored_feature, None)

    features['window_size'] = args.window_size

    if not args.windowizer:
        extractor = HandcraftedHashedFeaturesExtractor(
            n_features=args.hashed_features, return_features_dict=True,
            **features)
    else:
        extractor = WordWindowExtractor(args.window_size)

    if os.path.isfile(args.save_path):
        os.remove(args.save_path)
    elif os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)

    os.makedirs(args.save_path)

    instances = []
    instances_id = []
    features = []

    with open(args.corpus_labels, 'rb') as f:
        valid_lemmas = pickle.load(f)
        valid_lemmas = {lemma for lemma, (_, count) in
                        valid_lemmas[_LANGUAGE[args.corpus_language]]['lemmas'].items()
                        if count[count >= 3].shape[0] > 1}

    print('Getting instances', file=sys.stderr)

    corpora_files = sh.find(args.corpus, '-type', 'f').strip().split('\n')
    for corpus_file in tqdm(corpora_files):
        corpus_file = corpus_file.strip()
        parser = ColumnCorpusParser(corpus_file, *_CORPUS_COLUMNS[_LANGUAGE[args.corpus_language]])
        lemmas_for_corpus = {lemma: 0 for lemma in valid_lemmas}

        for sentence in tqdm(parser.sentences):
            for word in (word for word in sentence if word.tag.startswith('VM') and word.lemma in valid_lemmas
                         and lemmas_for_corpus[word.lemma] < int(args.max_instances / len(corpora_files)) + 1):
                lemmas_for_corpus[word.lemma] += 1
                sentence['main_lemma'] = word.lemma
                sentence['main_lemma_index'] = word.idx

                if args.windowizer:
                    instances.append(extractor.instantiate_sentence(sentence))
                else:
                    instance_features, instance = extractor.instantiate_sentence(sentence)
                    instances.append(instance)
                    features.append(instance_features)

                instances_id.append('%s:%s:%d:%s:%d' %
                                    (sentence.corpus_name, sentence.file, sentence.sentence_index,
                                     sentence.main_lemma, sentence.main_lemma_index))

    print('Saving resources in directory %s' % args.save_path, file=sys.stderr)

    if args.windowizer:
        matrix = np.array(instances)
    else:
        matrix = vstack(instances)

    if args.windowizer:
        np.savez_compressed(os.path.join(args.save_path, 'dataset.npz'),
                            data=matrix, instances_id=np.array(instances_id))
    else:
        np.savez_compressed(os.path.join(args.save_path, 'dataset.npz'),
                            data=matrix.data, indices=matrix.indices, indptr=matrix.indptr,
                            shape=matrix.shape, instances_id=np.array(instances_id))
        with open(os.path.join(args.save_path, 'features.p'), 'wb') as f:
            pickle.dump(features, f)

    print('Finished', file=sys.stderr)
