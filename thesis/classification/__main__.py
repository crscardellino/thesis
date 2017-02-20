# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import json
import numpy as np
import pandas as pd
import sys
import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from thesis.classification import BaselineClassifier, KerasMultilayerPerceptron
from thesis.dataset import SenseCorpusDatasets
from thesis.utils import try_number
from tqdm import tqdm


_CLASSIFIERS = {
    'baseline': BaselineClassifier,
    'decision_tree': DecisionTreeClassifier,
    'log': LogisticRegression,
    'mlp': KerasMultilayerPerceptron,
    'naive_bayes': MultinomialNB,
    'svm': LinearSVC
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

    if args.layers:
        config['layers'] = [args.layers] if isinstance(args.layers, int) else args.layers

    print('Loading data', file=sys.stderr)
    datasets = SenseCorpusDatasets(args.train_dataset, args.test_dataset)

    results = []

    print('Training models and getting results', file=sys.stderr)
    for lemma, data, target in tqdm(datasets.train_dataset.traverse_dataset_by_lemma(),
                                    total=datasets.train_dataset.num_lemmas):
        with tf.Graph().as_default() as g:  # To avoid resource exhaustion
            model = _CLASSIFIERS[args.classifier](**config)
            try:
                model.fit(data, target)
            except ValueError:  # Some classifiers cannot handle the case where the class is only one
                test_target = datasets.test_dataset.target(lemma)
                test_results = pd.DataFrame(np.vstack([test_target, test_target]).T,
                                            columns=['true', 'prediction'])
            else:
                test_data = datasets.test_dataset.data(lemma)
                test_target = datasets.test_dataset.target(lemma)
                test_results = pd.DataFrame(np.vstack([test_target, model.predict(test_data)]).T,
                                            columns=['true', 'prediction'])
            test_results.insert(0, 'lemma', lemma)
            results.append(test_results)
        del model
        del g

    print('Saving results to %s' % args.results_path, file=sys.stderr)
    pd.concat(results, ignore_index=True).to_csv(args.results_path, index=False)
