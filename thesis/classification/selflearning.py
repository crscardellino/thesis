#!/usr/bin/env bash

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import json
import os
import pandas as pd
import sys
import tensorflow as tf

from keras import backend as keras_backend
from thesis.classification.semisupervised import SelfLearningWrapper
from thesis.dataset import SenseCorpusDatasets, UnlabeledCorpusDataset
from thesis.dataset.utils import NotEnoughSensesError
from thesis.utils import try_number
from thesis.constants import CLASSIFIERS
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labeled_dataset_path')
    parser.add_argument('unlabeled_dataset_path')
    parser.add_argument('base_results_path')
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
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--acceptance_alpha', type=float, default=0.01)
    parser.add_argument('--error_sigma', type=float, default=0.1)
    parser.add_argument('--lemmas', nargs='+', default=set())
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--folds', type=int, default=0)
    parser.add_argument('--corpus_name', default='NA')
    parser.add_argument('--representation', default='NA')
    parser.add_argument('--vector_domain', default='NA')
    parser.add_argument('--predictions_only', action='store_true')

    args = parser.parse_args()

    if args.classifier_config_file is not None:
        with open(args.classifier_config_file, 'r') as fconfig:
            config = json.load(fconfig)
    else:
        config = {}

    if args.lemmas:
        args.lemmas = set(args.lemmas) if not isinstance(args.lemmas, set) else args.lemmas

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
    unlabeled_dataset_path = os.path.join(args.unlabeled_dataset_path, 'dataset.npz')
    unlabeled_features_path = os.path.join(args.unlabeled_dataset_path, 'features.p')
    unlabeled_dataset_extra_path = os.path.join(args.unlabeled_dataset_extra, 'dataset.npz') \
        if args.unlabeled_dataset_extra is not None else None

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

    print('Loading unlabeled dataset', file=sys.stderr)
    unlabeled_dataset = UnlabeledCorpusDataset(dataset_path=unlabeled_dataset_path,
                                               features_dict_path=unlabeled_features_path
                                               if args.word_vector_model_path is None else None,
                                               word_vector_model=labeled_datasets.train_dataset.word_vector_model,
                                               dataset_extra=unlabeled_dataset_extra_path)

    prediction_results = []
    certainty_progression = []
    features_progression = []
    cross_validation_results = []
    overfitting_measure_results = []
    results = (prediction_results, certainty_progression, features_progression,
               cross_validation_results, overfitting_measure_results)
    bootstrapped_instances = []
    bootstrapped_targets = []

    print('Running experiments per lemma', file=sys.stderr)
    for lemma, data, target, features in \
            tqdm(labeled_datasets.train_dataset.traverse_dataset_by_lemma(return_features=True),
                 total=labeled_datasets.train_dataset.num_lemmas):
        if not unlabeled_dataset.has_lemma(lemma):
            continue
        if args.lemmas and lemma not in args.lemmas:
            continue
        try:
            tf.reset_default_graph()
            tf.set_random_seed(args.random_seed)
            with tf.Session() as sess:
                keras_backend.set_session(sess)

                semisupervised = SelfLearningWrapper(
                    labeled_train_data=data, labeled_train_target=target,
                    labeled_test_data=labeled_datasets.test_dataset.data(lemma),
                    labeled_test_target=labeled_datasets.test_dataset.target(lemma),
                    unlabeled_data=unlabeled_dataset.data(lemma, limit=args.unlabeled_data_limit),
                    labeled_features=features, min_count=args.min_count, validation_ratio=args.validation_ratio,
                    acceptance_alpha=args.acceptance_alpha, random_seed=args.random_seed,
                    unlabeled_features=unlabeled_dataset.features_dictionaries(lemma, limit=args.unlabeled_data_limit),
                    candidates_limit=args.candidates_limit, error_sigma=args.error_sigma, lemma=lemma,
                    oversampling=True, predictions_only=args.predictions_only, overfitting_folds=args.folds)

                iterations = semisupervised.run(CLASSIFIERS[args.classifier], config)

                if iterations > 0:
                    for rst_agg, rst in zip(results, semisupervised.get_results()):
                        if rst is not None:
                            rst.insert(0, 'num_classes', semisupervised.classes.shape[0])
                            rst.insert(0, 'lemma', lemma)
                            rst.insert(0, 'layers', '_'.join(str(l) for l in args.layers) if args.layers else 'NA')
                            rst.insert(0, 'classifier', args.classifier)
                            rst.insert(0, 'algorithm', 'selflearning')
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
                    tqdm.write('Lemma %s - No iterations' % lemma, file=sys.stderr)
        except NotEnoughSensesError:
            tqdm.write('Lemma %s - Not enough senses with at least %d occurrences'
                       % (lemma, args.min_count), file=sys.stderr)
            continue

    print('Saving results', file=sys.stderr)

    try:
        if not args.predictions_only:
            pd.DataFrame({'instance': bootstrapped_instances, 'predicted_target': bootstrapped_targets}) \
                .to_csv('%s_unlabeled_dataset_predictions.csv' % args.base_results_path, index=False)
    except (ValueError, MemoryError) as e:
        print(e.args, file=sys.stderr)

    try:
        pd.concat(prediction_results, ignore_index=True) \
            .to_csv('%s_prediction_results.csv' % args.base_results_path, index=False, float_format='%.2e')
    except (ValueError, MemoryError) as e:
        print(e.args, file=sys.stderr)

    try:
        pd.concat(certainty_progression, ignore_index=True) \
            .to_csv('%s_certainty_progression.csv' % args.base_results_path, index=False, float_format='%.2e')
    except (ValueError, MemoryError) as e:
        print(e.args, file=sys.stderr)

    try:
        pd.concat(features_progression, ignore_index=True) \
            .to_csv('%s_features_progression.csv' % args.base_results_path, index=False, float_format='%.2e')
    except (ValueError, MemoryError) as e:
        print(e.args, file=sys.stderr)

    try:
        pd.concat(cross_validation_results, ignore_index=True) \
            .to_csv('%s_cross_validation_results.csv' % args.base_results_path, index=False, float_format='%.2e')
    except (ValueError, MemoryError) as e:
        print(e.args, file=sys.stderr)

    try:
        pd.concat(overfitting_measure_results, ignore_index=True) \
            .to_csv('%s_overfitting_measure_results.csv' % args.base_results_path, index=False, float_format='%.2e')
    except (ValueError, MemoryError) as e:
        print(e.args, file=sys.stderr)
