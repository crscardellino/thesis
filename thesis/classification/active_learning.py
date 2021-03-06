#!/usr/bin/env bash

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
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
    parser.add_argument('--full_senses_path', default=None)
    parser.add_argument('--sentences_path', default=None)
    parser.add_argument('--max_annotations', type=int, default=0)
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
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--candidates_selection', default='min')
    parser.add_argument('--error_sigma', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--folds', type=int, default=0)
    parser.add_argument('--annotation_lemmas', nargs='+', default=set())
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

    if args.annotation_lemmas:
        args.annotation_lemmas = set(args.annotation_lemmas) if not isinstance(args.annotation_lemmas, set) else\
            args.annotation_lemmas

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

    full_senses_dict = None

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

    if args.full_senses_path is not None:
        with open(args.full_senses_path, 'rb') as f:
            full_senses_dict = pickle.load(f)

    prediction_results = []
    certainty_progression = []
    features_progression = []
    cross_validation_results = []
    overfitting_measure_results = []
    senses = []
    results = (prediction_results, certainty_progression, features_progression,
               cross_validation_results, overfitting_measure_results)
    bootstrapped_instances = []
    bootstrapped_targets = []
    text_sentences = {}

    if args.sentences_path is not None:
        with open(args.sentences_path, 'r') as sfin:
            for line in sfin:
                iid, sent = line.strip().split('\t', 1)
                text_sentences[iid] = sent

    print('Running experiments per lemma', file=sys.stderr)
    for lemma, data, target, features in \
            tqdm(labeled_datasets.train_dataset.traverse_dataset_by_lemma(return_features=True),
                 total=labeled_datasets.train_dataset.num_lemmas):
        if unlabeled_dataset and not unlabeled_dataset.has_lemma(lemma):
            continue
        if args.annotation_lemmas and lemma not in args.annotation_lemmas:
            continue
        try:
            tf.reset_default_graph()
            with tf.Session() as sess:
                keras_backend.set_session(sess)

                if unlabeled_dataset is not None:
                    unlabeled_data = unlabeled_dataset.data(lemma, limit=args.unlabeled_data_limit)
                    unlabeled_target = np.array([])
                    unlabeled_features = unlabeled_dataset.features_dictionaries(lemma, limit=args.unlabeled_data_limit)
                    instances_id = unlabeled_dataset.instances_id(lemma, limit=args.unlabeled_data_limit)
                    lemma_unlabeled_sentences = [text_sentences[':'.join(iid)] for iid in instances_id]
                else:
                    li = np.in1d(labeled_datasets.train_dataset.lemmas_index(lemma), initial_indices)
                    ui = np.in1d(labeled_datasets.train_dataset.lemmas_index(lemma), unlabeled_indices)
                    unlabeled_data = data[ui]
                    unlabeled_target = target[ui]
                    unlabeled_features = list(compress(features, ui))
                    data = data[li]
                    target = target[li]
                    features = list(compress(features, li))
                    lemma_unlabeled_sentences = None

                test_data = labeled_datasets.test_dataset.data(lemma)
                test_target = labeled_datasets.test_dataset.target(lemma)

                _, zero_based_indices = np.unique(np.concatenate([target, unlabeled_target, test_target]),
                                                  return_inverse=True)
                train_classes = {label: idx for idx, label in
                                 enumerate(labeled_datasets.train_dataset.train_classes(lemma))}

                train_size = target.shape[0]
                unlabeled_size = unlabeled_target.shape[0]
                test_size = test_target.shape[0]

                target = zero_based_indices[:train_size]
                unlabeled_target = zero_based_indices[train_size:train_size+unlabeled_size]
                test_target = zero_based_indices[train_size+unlabeled_size:]

                semisupervised = ActiveLearningWrapper(
                    labeled_train_data=data, labeled_train_target=target, labeled_test_data=test_data,
                    labeled_test_target=test_target, unlabeled_data=unlabeled_data, unlabeled_target=unlabeled_target,
                    lemma=lemma, labeled_features=features, min_count=args.min_count, full_senses_dict=full_senses_dict,
                    validation_ratio=args.validation_ratio, candidates_selection=args.candidates_selection,
                    candidates_limit=args.candidates_limit, unlabeled_features=unlabeled_features,
                    error_sigma=args.error_sigma, folds=args.folds, random_seed=args.random_seed,
                    acceptance_threshold=0, unlabeled_sentences=lemma_unlabeled_sentences, train_classes=train_classes,
                    max_annotations=args.max_annotations
                )

                iterations = semisupervised.run(CLASSIFIERS[args.classifier], config)

                if iterations > 0:
                    for rst_agg, rst in zip(results, semisupervised.get_results()):
                        if rst is not None:
                            rst.insert(0, 'num_classes', semisupervised.classes.shape[0])
                            rst.insert(0, 'lemma', lemma)
                            rst.insert(0, 'candidates_selection', args.candidates_selection)
                            rst.insert(0, 'layers', '_'.join(str(l) for l in args.layers) if args.layers else 'NA')
                            rst.insert(0, 'classifier', args.classifier)
                            rst.insert(0, 'algorithm', 'active_learning')
                            rst.insert(0, 'vector_domain', args.vector_domain or 'NA')
                            rst.insert(0, 'representation', args.representation or 'NA')
                            rst.insert(0, 'corpus', args.corpus_name)
                            rst_agg.append(rst)

                    if args.full_senses_path is not None:
                        senses.append(semisupervised.get_senses())

                    # Save the bootstrapped data (if there is unlabeled data to save)
                    if unlabeled_dataset is not None:
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
        pd.DataFrame({'instance': bootstrapped_instances, 'predicted_target': bootstrapped_targets}) \
            .to_csv('%s_unlabeled_dataset_predictions.csv' % args.base_results_path, index=False, float_format='%.2e')
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
        pd.concat(senses, ignore_index=True) \
            .to_csv('%s_senses_description.csv' % args.base_results_path, index=False, float='%.2e')
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
