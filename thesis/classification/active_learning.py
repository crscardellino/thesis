#!/usr/bin/env bash

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import json
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

from itertools import compress
from keras import backend as keras_backend
from thesis.classification.semisupervised import ActiveLearningWrapper
from thesis.dataset import SenseCorpusDatasets, UnlabeledCorpusDataset
from thesis.dataset.utils import NotEnoughSensesError
from thesis.utils import try_number
from thesis.constants import CLASSIFIERS
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labeled_dataset_path')
    parser.add_argument('base_results_path')
    parser.add_argument('--unlabeled_dataset_path', default=None)
    parser.add_argument('--simulation_indices_path', default=None)
    parser.add_argument('--word_vector_model_path', default=None)
    parser.add_argument('--labeled_dataset_extra', default=None)
    parser.add_argument('--unlabeled_dataset_extra', default=None)
    parser.add_argument('--classifier', type=str, default='svm')
    parser.add_argument('--classifier_config_file', type=str, default=None)
    parser.add_argument('--classifier_config', type=lambda config: tuple(config.split('=')),
                        default=list(), nargs='+')
    parser.add_argument('--layers', type=int, nargs='+', default=list())
    parser.add_argument('--unlabeled_data_limit', type=int, default=1000)
    parser.add_argument('--candidates_limit', type=int, default=0)
    parser.add_argument('--min_count', type=int, default=2)
    parser.add_argument('--validation_ratio', type=float, default=0.1)
    parser.add_argument('--candidates_selection', default='min')
    parser.add_argument('--error_sigma', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--folds', type=int, default=0)
    parser.add_argument('--corpus_name', default='NA')
    parser.add_argument('--representation', default='NA')
    parser.add_argument('--vector_domain', default='NA')

    args = parser.parse_args()

    if (args.unlabeled_dataset_path is None) == (args.simulation_indices_path is None):
        print('Either give an unlabeled dataset path or a simulation indices path', file=sys.stderr)
        sys.exit(1)

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
        args.layers = [args.layers] if isinstance(args.layers, int) else args.layers
        config['layers'] = args.layers

    labeled_datasets_path = os.path.join(args.labeled_dataset_path, '%s_dataset.npz')
    labeled_features_path = os.path.join(args.labeled_dataset_path, '%s_features.p')
    labeled_datasets_extra_path = os.path.join(args.labeled_dataset_extra, '%s_dataset.npz') \
        if args.labeled_dataset_extra is not None else None

    print('Loading labeled dataset', file=sys.stderr)
    labeled_datasets = SenseCorpusDatasets(train_dataset_path=labeled_datasets_path % 'train',
                                           train_features_dict_path=labeled_features_path % 'train'
                                           if args.word_vector_model_path is None else None,
                                           test_dataset_path=labeled_datasets_path % 'test',
                                           test_features_dict_path=labeled_features_path % 'test'
                                           if args.word_vector_model_path is None else None,
                                           word_vector_model_path=args.word_vector_model_path,
                                           train_dataset_extra=labeled_datasets_extra_path % 'train'
                                           if labeled_datasets_extra_path is not None else None,
                                           test_dataset_extra=labeled_datasets_extra_path % 'test'
                                           if labeled_datasets_extra_path is not None else None)

    if args.unlabeled_dataset_path:
        unlabeled_dataset_path = os.path.join(args.unlabeled_dataset_path, 'dataset.npz')
        unlabeled_features_path = os.path.join(args.unlabeled_dataset_path, 'features.p')
        unlabeled_dataset_extra_path = os.path.join(args.unlabeled_dataset_extra, 'dataset.npz') \
            if args.unlabeled_dataset_extra is not None else None

        print('Loading unlabeled dataset', file=sys.stderr)
        unlabeled_dataset = UnlabeledCorpusDataset(dataset_path=unlabeled_dataset_path,
                                                   features_dict_path=unlabeled_features_path
                                                   if args.word_vector_model_path is None else None,
                                                   word_vector_model=labeled_datasets.train_dataset.word_vector_model,
                                                   dataset_extra=unlabeled_dataset_extra_path)
        initial_indices = None
        unlabeled_indices = None
    else:
        simulation_indices = np.load(args.simulation_indices_path)
        initial_indices = simulation_indices['initial_indices']
        unlabeled_indices = simulation_indices['unlabeled_indices']
        unlabeled_dataset = None

    prediction_results = []
    certainty_progression = []
    features_progression = []
    cross_validation_results = []
    results = (prediction_results, certainty_progression, features_progression, cross_validation_results)
    bootstrapped_instances = []
    bootstrapped_targets = []

    print('Running experiments per lemma', file=sys.stderr)
    for lemma, data, target, features in \
            tqdm(labeled_datasets.train_dataset.traverse_dataset_by_lemma(return_features=True),
                 total=labeled_datasets.train_dataset.num_lemmas):
        if unlabeled_dataset and not unlabeled_dataset.has_lemma(lemma):
            continue
        try:
            tf.reset_default_graph()
            with tf.Session() as sess:
                keras_backend.set_session(sess)

                if unlabeled_dataset:
                    unlabeled_data = unlabeled_dataset.data(lemma, limit=args.unlabeled_data_limit)
                    unlabeled_target = None
                    unlabeled_features = unlabeled_dataset.features_dictionaries(lemma, limit=args.unlabeled_data_limit)
                else:
                    li = np.in1d(labeled_datasets.train_dataset.lemmas_index(lemma), initial_indices)
                    ui = np.in1d(labeled_datasets.train_dataset.lemmas_index(lemma), unlabeled_indices)
                    unlabeled_data = data[ui]
                    unlabeled_target = target[ui]
                    unlabeled_features = list(compress(features, ui))
                    data = data[li]
                    target = target[li]
                    features = list(compress(features, li))

                semisupervised = ActiveLearningWrapper(
                    labeled_train_data=data, labeled_train_target=target,
                    labeled_test_data=labeled_datasets.test_dataset.data(lemma),
                    labeled_test_target=labeled_datasets.test_dataset.target(lemma),
                    unlabeled_data=unlabeled_data, unlabeled_target=unlabeled_target,
                    labeled_features=features, min_count=args.min_count, validation_ratio=args.validation_ratio,
                    candidates_selection=args.candidates_selection, candidates_limit=args.candidates_limit,
                    unlabeled_features=unlabeled_features, error_sigma=args.error_sigma, folds=args.folds,
                    random_seed=args.random_seed)

                if semisupervised.run(CLASSIFIERS[args.classifier], config) > 0:
                    for rst_agg, rst in zip(results, semisupervised.get_results()):
                        if rst:
                            rst.insert(0, 'folds', args.folds if args.folds > 0 else 'NA')
                            rst.insert(0, 'num_classes', semisupervised.classes.shape[0])
                            rst.insert(0, 'lemma', lemma)
                            rst.insert(0, 'candidates_limit', args.candidates_limit)
                            rst.insert(0, 'candidates_selection', args.candidates_selection)
                            rst.insert(0, 'layers', '_'.join(str(l) for l in args.layers) if args.layers else 'NA')
                            rst.insert(0, 'classifier', args.classifier)
                            rst.insert(0, 'vector_domain', args.vector_domain or 'NA')
                            rst.insert(0, 'representation', args.representation or 'NA')
                            rst.insert(0, 'corpus', args.corpus_name)
                            rst_agg.append(rst)

                    # Save the bootstrapped data
                    bi, bt = semisupervised.bootstrapped()
                    bootstrapped_targets.extend(bt)

                    ul_instances = unlabeled_dataset.instances_id(lemma, limit=args.unlabeled_data_limit)
                    bootstrapped_instances.extend(':'.join(ul_instances[idx]) for idx in bi)
                else:
                    tqdm.write('The lemma %s didn\'t run iterations' % lemma, file=sys.stderr)
        except NotEnoughSensesError:
            tqdm.write('The lemma %s doesn\'t have enough senses with at least %d occurrences'
                       % (lemma, args.min_count), file=sys.stderr)
            continue

    print('Saving results', file=sys.stderr)

    pd.DataFrame({'instance': bootstrapped_instances, 'predicted_target': bootstrapped_targets}) \
        .to_csv('%s_unlabeled_dataset_predictions.csv' % args.base_results_path, index=False)
    pd.concat(prediction_results, ignore_index=True) \
        .to_csv('%s_prediction_results.csv' % args.base_results_path, index=False, float_format='%d')
    pd.concat(certainty_progression, ignore_index=True) \
        .to_csv('%s_certainty_progression.csv' % args.base_results_path, index=False, float_format='%.2e')
    pd.concat(features_progression, ignore_index=True) \
        .to_csv('%s_features_progression.csv' % args.base_results_path, index=False, float_format='%d')

    if cross_validation_results:
        pd.concat(cross_validation_results, ignore_index=True) \
            .to_csv('%s_cross_validation_results.csv' % args.base_results_path, index=False, float_format='%d')
