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

from keras import backend as K
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from thesis.classification import BaselineClassifier, KerasMultilayerPerceptron
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


def folds_training(folds, splits, data, target, lemma, classifier, config):
    cummulative_indexes = []
    return_results = []

    for split_index, indices in enumerate(idxs for _, idxs in StratifiedKFold(splits).split(data, target)):
        cummulative_indexes.extend(indices)

        train_data = data[cummulative_indexes]
        train_target = target[cummulative_indexes]

        kf = StratifiedKFold(folds)
        for fold_index, (train_indices, test_indices) in enumerate(kf.split(train_data, train_target)):
            tf.reset_default_graph()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                K.set_session(sess)

                fold_train_data = train_data[train_indices]
                fold_test_data = train_data[test_indices]
                fold_train_target = train_target[train_indices]
                fold_test_target = train_target[test_indices]
                labels = np.unique(fold_train_target)

                try:
                    model = _CLASSIFIERS[classifier](**config)
                    model.fit(fold_train_data, fold_train_target)
                except ValueError:
                    if np.unique(fold_train_target).shape[0] != 1:
                        raise
                    model = BaselineClassifier()
                    model.fit(fold_train_data, fold_train_target)

                try:
                    train_error = log_loss(fold_train_target, model.predict_proba(fold_train_data), labels=labels)
                    test_error = log_loss(fold_test_target, model.predict_proba(fold_test_data), labels=labels)
                except ValueError:
                    if np.unique(fold_train_target).shape[0] != 1:
                        raise
                    train_error = 0
                    test_error = 0

                fold_train_results = pd.DataFrame(
                    np.vstack([fold_train_target, model.predict(fold_train_data)]).T,
                    columns=['true', 'prediction']
                )
                fold_train_results.insert(0, 'error', train_error)
                fold_train_results.insert(0, 'fold', 'train.%d' % fold_index)

                fold_test_results = pd.DataFrame(
                    np.vstack([fold_test_target, model.predict(fold_test_data)]).T,
                    columns=['true', 'prediction']
                )
                fold_test_results.insert(0, 'error', test_error)
                fold_test_results.insert(0, 'fold', 'test.%d' % fold_index)

                fold_results = pd.concat([fold_train_results, fold_test_results], ignore_index=True)
                fold_results.insert(0, 'size', indices.shape[0])
                fold_results.insert(0, 'split', split_index)
                fold_results.insert(0, 'lemma', lemma)

                return_results.append(fold_results)

    return return_results


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
                        default=1,
                        type=int,
                        help='Number of splits for increasing corpus size in training.')
    parser.add_argument('--folds',
                        default=0,
                        type=int,
                        help='Activate to run the different folds.')
    parser.add_argument('--ensure_minimum',
                        action='store_true',
                        help='In case of using folds ensure the minimum amount of classes needed is respected.')

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

    if args.layers:
        config['layers'] = [args.layers] if isinstance(args.layers, int) else args.layers

    print('Loading data', file=sys.stderr)
    datasets = SenseCorpusDatasets(args.train_dataset, args.test_dataset)

    results = []

    print('Training models and getting results', file=sys.stderr)
    for lemma, data, target in tqdm(datasets.train_dataset.traverse_dataset_by_lemma(),
                                    total=datasets.train_dataset.num_lemmas):
        if args.folds > 0:
            # We use everything but removing classes we have no idea about (-1)
            data = sps.vstack([data, datasets.test_dataset.data(lemma)])
            target = np.concatenate([target, datasets.test_dataset.target(lemma)])

            filtered = np.where(target != -1)[0]

            data = data[filtered]
            target = target[filtered]

            if args.ensure_minimum:
                labels, counts = np.unique(target, return_counts=True)
                minimum_counts = np.where(counts >= args.folds * args.splits)[0]

                if minimum_counts.shape[0] < 2:
                    print('Lemma %s has no sufficient classes to ensure minimum' % lemma, file=sys.stderr)
                    continue

                filtered_by_count = np.in1d(target, labels[minimum_counts])
                data = data[filtered_by_count]
                target = target[filtered_by_count]

            if 0 < args.max_features < datasets.train_dataset.input_vector_size():
                selector.fit(data, target)
                data = selector.transform(data)

            results.extend(folds_training(args.folds, args.splits, data, target, lemma, args.classifier, config))
        else:
            tf.reset_default_graph()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                K.set_session(sess)
                selector = SelectKBest(_FEATURE_SELECTION[args.feature_selection], k=args.max_features)

                if 0 < args.max_features < datasets.train_dataset.input_vector_size():
                    selector.fit(data, target)
                    data = selector.transform(data)

                model = _CLASSIFIERS[args.classifier](**config)

                try:
                    model.fit(data, target)
                except ValueError:
                    # Some classifiers cannot handle the case where the class is only one
                    # In that case we use the baseline of most frequent class
                    if np.unique(target).shape[0] != 1:
                        raise
                    model = BaselineClassifier()
                    model.fit(data, target)

                train_results = pd.DataFrame(np.vstack([target, model.predict(data)]).T,
                                             columns=['true', 'prediction'])
                train_results.insert(0, 'corpus', 'train')

                test_data = datasets.test_dataset.data(lemma)
                test_target = datasets.test_dataset.target(lemma)

                if 0 < args.max_features < datasets.train_dataset.input_vector_size():
                    test_data = selector.transform(test_data)

                test_results = pd.DataFrame(np.vstack([test_target, model.predict(test_data)]).T,
                                            columns=['true', 'prediction'])
                test_results.insert(0, 'corpus', 'test')

                all_results = pd.concat([train_results, test_results], ignore_index=True)
                all_results.insert(0, 'lemma', lemma)

                results.append(all_results)

    print('Saving results to %s' % args.results_path, file=sys.stderr)
    pd.concat(results, ignore_index=True).to_csv(args.results_path, index=False, float_format='%.2e')
