# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import numpy as np
import os
import shutil
import sys

from collections import defaultdict
from scipy.sparse import vstack
from sklearn.feature_extraction import DictVectorizer
from thesis.feature_extraction import HandcraftedFeaturesExtractor, HandcraftedHashedFeaturesExtractor
from thesis.parsers import ColumnCorpusParser
from tqdm import tqdm


DEFAULT_FEATURES = {
    'main_token': True, 'main_lemma': True, 'main_tag': True, 'main_morpho': True,
    'window_bow': True, 'window_tokens': True, 'window_lemmas': True, 'window_tags': True,
    'surrounding_bigrams': True, 'surrounding_trigrams': True,
    'inbound_dep_triples': True, 'outbound_dep_triples': True
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('corpus',
                        type=str,
                        help='Path to the corpus file to featurize.')
    parser.add_argument('save_path',
                        type=str,
                        help='Path to directory to save the matrices. Will be overwritten if exists.')
    parser.add_argument('columns',
                        type=str,
                        nargs='+',
                        help='Columns in the corpus file.')
    parser.add_argument('--total_sentences',
                        type=int,
                        default=0,
                        help='Number of sentences to parse.')
    parser.add_argument('--hashing',
                        action='store_true',
                        help='If active use the `hashing trick` for featurization.')
    parser.add_argument('--hashed_features',
                        type=int,
                        default=2**10,
                        help='In case of using hashing trick, number of features after hashing (default: 1024).')
    parser.add_argument('--negative_hash',
                        action='store_true',
                        help='Whether to use or not negative features when hashing.')
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

    args.columns = [args.columns] if isinstance(args.columns, str) else args.columns
    args.ignored_features = [args.ignored_features]\
        if isinstance(args.ignored_features, str) else args.ignored_features

    features = DEFAULT_FEATURES.copy()

    for ignored_feature in args.ignored_features:
        features.pop(ignored_feature, None)

    features['window_size'] = args.window_size

    extractor = HandcraftedHashedFeaturesExtractor(
        n_features=args.hashed_features,
        non_negative=not args.negative_hash,
        **features) if args.hashing else HandcraftedFeaturesExtractor(**features)

    parser = ColumnCorpusParser(args.corpus, *args.columns)

    if os.path.isfile(args.save_path):
        os.remove(args.save_path)
    elif os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)

    os.makedirs(args.save_path)

    instances = defaultdict(list)
    labels = defaultdict(list)
    sentences_id = defaultdict(list)

    print('Getting instances', file=sys.stderr)

    for sentence in tqdm(parser.sentences, total=args.total_sentences):
        if sentence.corpus == 'filtered':
            continue

        main_lemma_tag = getattr(sentence, 'lemma_tag', 'v')
        corpus = 'nonverb.%s' % sentence.corpus if main_lemma_tag != 'v' else sentence.corpus
        label = '%s.%s' % (sentence.main_lemma, sentence.sense)

        instances[corpus].append(extractor.featurize_sentence(sentence))
        labels[corpus].append(label)
        sentences_id[corpus].append(sentence.sentence_index)

    print('Saving resources in directory %s' % args.save_path, file=sys.stderr)

    # The train classes define the rest of the labels
    train_classes = {lbl: idx for idx, lbl in enumerate(labels['train'])}

    for corpus in instances:
        if args.hashing:
            matrix = vstack(instances[corpus])
        else:
            vectorizer = DictVectorizer()
            matrix = vectorizer.fit_transform(instances[corpus])

        classes = np.unique(labels[corpus])
        target = np.array([train_classes.get(lbl, -1) for lbl in labels[corpus]])
        sentences = np.array(sentences_id[corpus])
        np.savez_compressed(os.path.join(args.save_path, '%s_dataset.npz' % corpus),
                            data=matrix.data, indices=matrix.indices, indptr=matrix.indptr, shape=matrix.shape,
                            target=target, classes=sorted(train_classes), corpus_classes=classes,
                            sentences=sentences)

    print('Finished', file=sys.stderr)
