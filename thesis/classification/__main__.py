# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import json
import numpy as np
import pandas as pd
import scipy.sparse as sps
import sys
import tensorflow as tf
import warnings

from keras import backend as keras_backend
from scipy.sparse import issparse
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from thesis.classification import BaselineClassifier, KerasMultilayerPerceptron, learning_curve_training
from thesis.dataset import SenseCorpusDatasets
from thesis.utils import try_number
from tqdm import tqdm

# Set logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


_CLASSIFIERS = {
    'baseline': BaselineClassifier,
    'decision_tree': DecisionTreeClassifier,
    'log': LogisticRegression,
    'mlp': KerasMultilayerPerceptron,
    'naive_bayes': MultinomialNB,
    'svm': SVC
}

_FEATURE_SELECTION = {
    'chi2': chi2,
    'f_classif': f_classif,
    'mutual_info_classif': mutual_info_classif
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('train_dataset',
                        type=str,
                        help='Path to train dataset file.')
    parser.add_argument('test_dataset',
                        type=str,
                        help='Path to test dataset file.')
    parser.add_argument('results_path',
                        type=str,
                        help='Path to csv to store test prediction results.')
    parser.add_argument('--classifier',
                        type=str,
                        default='decision_tree',
                        help='Classification algorithm to use (default: decision_tree).')
    parser.add_argument('--classifier_config_file',
                        type=str,
                        default=None,
                        help='Path to the configuration file.')
    parser.add_argument('--classifier_config',
                        type=lambda config: tuple(config.split('=')),
                        default=list(),
                        nargs='+',
                        help='Classifier manual configuration (will override the config file).')
    parser.add_argument('--layers',
                        type=int,
                        nargs='+',
                        default=list(),
                        help='Layers for multilayer perceptron.')
    parser.add_argument('--max_features',
                        type=int,
                        default=0,
                        help='Max features to train the classifier with (needs a feature selection method).')
    parser.add_argument('--feature_selection',
                        default='f_classif',
                        help='Feature selection method to apply to the dataset.')
    parser.add_argument('--splits',
                        default=0,
                        type=int,
                        help='Number of splits for increasing corpus size in training.')
    parser.add_argument('--folds',
                        default=0,
                        type=int,
                        help='Activate to run the different folds.')
    parser.add_argument('--word_vectors_model_path',
                        type=str,
                        default=None,
                        help='Path to the word vectors file in case of using one.')
    parser.add_argument('--train_dataset_extra',
                        type=str,
                        default=None,
                        help='If given, it uses the train dataset as an extra to merge with ' +
                             'the original datset (useful to add word vectors to handcrafted features)')
    parser.add_argument('--test_dataset_extra',
                        type=str,
                        default=None,
                        help='If given, it uses the test dataset as an extra to merge with ' +
                             'the original datset (useful to add word vectors to handcrafted features)')
    parser.add_argument('--min_count',
                        type=int,
                        default=2,
                        help='Minimum number of per-class occurrences.')
    parser.add_argument('--corpus_name',
                        default='sensem',
                        type=str,
                        help='Name of the corpus.')
    parser.add_argument('--representation',
                        default=None,
                        type=str,
                        help='Type of representation.')
    parser.add_argument('--vector_domain',
                        default=None,
                        type=str,
                        help='Vector domain.')

    args = parser.parse_args()

    print('Loading classifier configuration', file=sys.stderr)
    if args.classifier_config_file is not None:
        with open(args.classifier_config_file, 'r') as fconfig:
            config = json.load(fconfig)
    else:
        config = {}

    args.classifier_config = [args.classifier_config] \
        if isinstance(args.classifier_config, tuple) else args.classifier_config

    for key, value in args.classifier_config:
        config[key] = try_number(value)

    if args.classifier == 'svm':
        config['kernel'] = 'linear'
        config['probability'] = True

    args.layers = [args.layers] if isinstance(args.layers, int) else args.layers
    if args.layers:
        config['layers'] = args.layers

    print('Loading data', file=sys.stderr)
    datasets = SenseCorpusDatasets(args.train_dataset, args.test_dataset,
                                   word_vector_model_path=args.word_vectors_model_path,
                                   train_dataset_extra=args.train_dataset_extra,
                                   test_dataset_extra=args.test_dataset_extra)

    results = []
    discarded_lemmas = []

    print('Training models and getting results', file=sys.stderr)
    for lemma, data, target in tqdm(datasets.train_dataset.traverse_dataset_by_lemma(),
                                    total=datasets.train_dataset.num_lemmas):
        selector = SelectKBest(_FEATURE_SELECTION[args.feature_selection], k=args.max_features)

        if args.folds > 0:
            # We use everything but removing classes we have no idea about (-1)
            data = sps.vstack([data, datasets.test_dataset.data(lemma)])\
                if issparse(data) else np.vstack((data, datasets.test_dataset.data(lemma)))
            target = np.concatenate([target, datasets.test_dataset.target(lemma)])

            filtered = np.where(target != -1)[0]
            data = data[filtered]
            target = target[filtered]

            labels, counts = np.unique(target, return_counts=True)
            minimum_counts = np.where(counts >= args.min_count)[0]

            if minimum_counts.shape[0] < 2:
                tqdm.write('Lemma %s has no sufficient classes to ensure minimum count' % lemma, file=sys.stderr)
                discarded_lemmas.append(lemma)
                continue

            filtered_by_count = np.in1d(target, labels[minimum_counts])
            data = data[filtered_by_count]
            target = target[filtered_by_count]

            if 0 < args.max_features < datasets.train_dataset.input_vector_size():
                selector.fit(data, target)
                data = selector.transform(data)

            for learning_curve_results in learning_curve_training(_CLASSIFIERS[args.classifier], data, target,
                                                                  args.classifier_config, args.folds, args.splits,
                                                                  args.min_count):
                learning_curve_results.insert(0, 'num_classes', minimum_counts.shape[0])
                learning_curve_results.insert(0, 'lemma', lemma)
                learning_curve_results.insert(0, 'layers',
                                              '_'.join(str(l) for l in args.layers) if args.layers else 'NA')
                learning_curve_results.insert(0, 'classifier', args.classifier)
                learning_curve_results.insert(0, 'vector_domain', args.vector_domain or 'NA')
                learning_curve_results.insert(0, 'representation', args.representation or 'NA')
                learning_curve_results.insert(0, 'corpus', args.corpus_name)

                results.append(learning_curve_results)
        else:
            tf.reset_default_graph()
            with tf.Session() as sess:
                keras_backend.set_session(sess)

                labels = np.unique(target)

                if labels.shape[0] < 2:
                    tqdm.write('Lemma %s has no sufficient classes' % lemma, file=sys.stderr)
                    discarded_lemmas.append(lemma)
                    continue

                if 0 < args.max_features < datasets.train_dataset.input_vector_size():
                    selector.fit(data, target)
                    data = selector.transform(data)

                model = _CLASSIFIERS[args.classifier](**config)
                model.fit(data, target)

                train_results = pd.DataFrame(
                    {'true': target.astype(np.int32),
                     'prediction': model.predict(data).astype(np.int32)},
                    columns=['true', 'prediction'])
                train_results.insert(0, 'corpus_split', 'train')

                test_data = datasets.test_dataset.data(lemma)
                test_target = datasets.test_dataset.target(lemma)

                if 0 < args.max_features < datasets.train_dataset.input_vector_size():
                    test_data = selector.transform(test_data)

                test_results = pd.DataFrame(
                    {'true': test_target.astype(np.int32),
                     'prediction': model.predict(test_data).astype(np.int32)},
                    columns=['true', 'prediction'])
                test_results.insert(0, 'corpus_split', 'test')

                lemma_results = pd.concat([train_results, test_results], ignore_index=True)
                lemma_results.insert(0, 'num_classes', labels.shape[0])
                lemma_results.insert(0, 'lemma', lemma)
                lemma_results.insert(0, 'layers', '_'.join(str(l) for l in args.layers) if args.layers else 'NA')
                lemma_results.insert(0, 'classifier', args.classifier)
                lemma_results.insert(0, 'vector_domain', args.vector_domain or 'NA')
                lemma_results.insert(0, 'representation', args.representation or 'NA')
                lemma_results.insert(0, 'corpus', args.corpus_name)

                results.append(lemma_results)

    print('There was a total of %d out of %d lemmas discarded' %
          (len(discarded_lemmas), datasets.train_dataset.num_lemmas),
          file=sys.stderr)
    print('The discarded lemmas are:\n%s' % ', '.join(discarded_lemmas), file=sys.stderr)

    try:
        print('Saving results to %s' % args.results_path, file=sys.stderr)
        pd.concat(results, ignore_index=True).to_csv(args.results_path, index=False, float_format='%.2f')
    except ValueError:
        print('No results to save', file=sys.stderr)
