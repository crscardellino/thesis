# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import argparse
import json
import numpy as np
import os
import pandas as pd
import scipy.sparse as sps
import sys
import tensorflow as tf

from sklearn.metrics import log_loss
from sklearn.svm import SVC
from thesis.classification import KerasMultilayerPerceptron
from thesis.dataset import SenseCorpusDatasets, UnlabeledCorpusDataset
from thesis.dataset.utils import filter_minimum, validation_split
from thesis.utils import try_number, RANDOM_SEED
from tqdm import tqdm


_CLASSIFIERS = {
    'mlp': KerasMultilayerPerceptron,
    'svm': SVC
}


def _feature_transformer(feature):
    if isinstance(feature[1], str):
        return '='.join(feature), 1
    else:
        return feature


class SemiSupervisedWrapper(object):
    def __init__(self, labeled_train_data, labeled_train_target, labeled_test_data, labeled_test_target,
                 unlabeled_data, labeled_features, unlabeled_features, min_count=2, validation_ratio=0.1,
                 acceptance_threshold=0.8, candidates_selection='max', candidates_limit=0, error_sigma=0.001,
                 random_seed=RANDOM_SEED):
        filtered_values = filter_minimum(target=labeled_train_target, min_count=min_count)
        train_index, validation_index = validation_split(target=labeled_train_target[filtered_values],
                                                         validation_ratio=validation_ratio, random_seed=random_seed)

        self._labeled_train_data = labeled_train_data[filtered_values][train_index]
        self._labeled_train_target = labeled_train_target[filtered_values][train_index]
        self._labeled_validation_data = labeled_train_data[filtered_values][validation_index]
        self._labeled_validation_target = labeled_train_target[filtered_values][validation_index]
        self._labeled_test_data = labeled_test_data
        self._labeled_test_target = labeled_test_target
        self._unlabeled_data = unlabeled_data

        # Features dictionaries
        labeled_features = [lf for idx, lf in enumerate(labeled_features) if idx in set(filtered_values)]
        labeled_features = [lf for idx, lf in enumerate(labeled_features) if idx in set(train_index)]
        self._labeled_features = labeled_features
        self._unlabeled_features = unlabeled_features

        self._labels = np.unique(self._labeled_train_target)
        self._bootstrapped_indices = []
        self._bootstrapped_targets = []
        self._model = None

        self._prediction_results = []
        self._error_progression = []
        self._error_sigma = error_sigma
        self._features_progression = []
        self._certainty_progression = []

        self._acceptance_threshold = acceptance_threshold
        self._candidates_selection = candidates_selection
        self._candidates_limit = candidates_limit

    def _get_candidates(self, prediction_probabilities):
        # Get the max probabilities per target
        max_probabilities = prediction_probabilities.max(axis=1)

        # Sort the candidate probabilities according to the selection method
        if self._candidates_selection == 'min':
            candidates = max_probabilities.argsort()
        elif self._candidates_selection == 'max':
            candidates = max_probabilities.argsort()[::-1]
        elif self._candidates_selection == 'random':
            candidates = np.random.permutation(max_probabilities.shape[0])
        else:
            raise ValueError('Not a valid candidate selection method: %s' % self._candidates_selection)

        # Select the candidates, limiting them in case of an existing limit
        if self._candidates_limit > 0:
            candidates = candidates[:self._candidates_limit]

        # If there is an acceptance threshold filter out candidates that doesn't comply it
        if self._acceptance_threshold > 0:
            over_threshold = np.where(max_probabilities[candidates] >= self._acceptance_threshold)[0]
            candidates = candidates[over_threshold]

        return candidates

    def _add_results(self, corpus_split, iteration):
        # Get the labeled data/target
        data = getattr(self, '_labeled_%s_data' % corpus_split)
        target = getattr(self, '_labeled_%s_target' % corpus_split)

        # For train corpus we need to append the bootstrapped data and targets
        if corpus_split == 'train':
            data = (data, self._unlabeled_data[self._bootstrapped_indices])
            data = sps.vstack(data) if sps.issparse(self._labeled_train_data) else np.vstack(data)
            target = np.concatenate((target, self._bootstrapped_targets))

            # Add the features of the new data to the progression
            unlabeled_features = [uf for idx, uf in enumerate(self._unlabeled_features)
                                  if idx in set(self._bootstrapped_indices)]
            features = self._labeled_features + unlabeled_features

            for tgt, feats in zip(target, features):
                feats = [_feature_transformer(f) for f in sorted(feats.items())]
                fdf = pd.DataFrame(feats, columns=['feature', 'count'])
                fdf.insert(0, 'target', tgt)
                fdf.insert(0, 'iteration', iteration)

                self._features_progression.append(fdf)

        # Calculate cross entropy error (perhaps better with the algorithm by itself)
        # and update the results of the iteration giving the predictions
        error = log_loss(target, self._model.predict_proba(data), labels=self._labels)
        results = pd.DataFrame(np.vstack([target, self._model.predict(data)]).T,
                               columns=['true', 'prediction'])
        results.insert(0, 'error', error)
        results.insert(0, 'iteration', iteration)
        results.insert(0, 'corpus_split', corpus_split)

        # For the validation corpus, we append the error progression
        if corpus_split == 'validation':
            self._error_progression.append(error)

        # Add the results to the corresponding corpus split results
        self._prediction_results.append(results)

    def bootstrapped(self):
        return self._bootstrapped_indices, self._bootstrapped_targets

    def get_results(self):
        prediction_results = pd.concat(self._prediction_results, ignore_index=True)
        certainty_progression = pd.concat(self._certainty_progression, ignore_index=True)
        features_progression = pd.concat(self._features_progression, ignore_index=True)

        return prediction_results, certainty_progression, features_progression

    def run(self, model_class, model_config):
        self._model = model_class(**model_config)
        self._model.fit(self._labeled_train_data, self._labeled_train_target)

        for corpus_split in ('train', 'validation', 'test'):
            self._add_results(corpus_split, 'initial')

        iteration = 0
        bootstrap_mask = np.ones(self._unlabeled_data.shape[0], dtype=np.bool)

        while len(self._bootstrapped_indices) < self._unlabeled_data.shape[0]:
            bootstrap_mask[self._bootstrapped_indices] = False
            prediction_probabilities = self._model.predict_proba(self._unlabeled_data[bootstrap_mask])

            candidates = self._get_candidates(prediction_probabilities)
            data_candidates = self._unlabeled_data[candidates]
            target_candidates = prediction_probabilities[candidates].argmax(axis=1)

            train_data = (self._labeled_train_data, self._unlabeled_data[self._bootstrapped_indices], data_candidates)
            train_data = sps.vstack(train_data) if sps.issparse(self._labeled_train_data) else np.vstack(train_data)
            train_target = np.concatenate((self._labeled_train_target, self._bootstrapped_targets, target_candidates))

            # Train the new model
            new_model = model_class(**model_config)
            new_model.fit(train_data, train_target)

            validation_error = log_loss(self._labeled_validation_target,
                                        new_model.predict_proba(self._labeled_validation_data),
                                        labels=self._labels)

            if self._error_sigma > 0 and validation_error > self._error_progression[-1] + self._error_sigma:
                break

            self._model = new_model
            self._bootstrapped_indices.extend(candidates)
            self._bootstrapped_targets.extend(target_candidates)
            iteration += 1

            # Add the certainty of the predicted classes of the unseen examples to the certainty progression results
            certainty_df = pd.DataFrame(prediction_probabilities.max(axis=1), columns=['certainty'])
            certainty_df.insert(0, 'iteration', iteration)
            self._certainty_progression.append(certainty_df)

            for corpus_split in ('train', 'validation'):
                self._add_results(corpus_split, iteration)

        for corpus_split in ('train', 'test'):
            self._add_results(corpus_split, 'final')


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
    parser.add_argument('--validation_ratio', type=float, default=0.1)
    parser.add_argument('--acceptance_threshold', type=float, default=0.9)
    parser.add_argument('--candidates_selection', default='max')
    parser.add_argument('--error_sigma', type=float, default=0.001)
    parser.add_argument('--random_seed', type=int, default=1234)

    args = parser.parse_args()

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

    labeled_dataset = os.path.join(args.labeled_dataset_path, '%s_dataset.npz')
    labeled_features = os.path.join(args.labeled_dataset_path, '%s_features.p')
    labeled_extra = os.path.join(args.labeled_dataset_extra, '%s_dataset.npz')\
        if args.labeled_dataset_extra is not None else None
    unlabeled_dataset = os.path.join(args.unlabeled_dataset_path, 'dataset.npz')
    unlabeled_features = os.path.join(args.unlabeled_dataset_path, 'features.p')
    unlabeled_dataset_extra = os.path.join(args.unlabeled_dataset_extra, 'dataset.npz')\
        if args.unlabeled_dataset_extra is not None else None

    print('Loading labeled dataset', file=sys.stderr)
    labeled_datasets = SenseCorpusDatasets(train_dataset_path=labeled_dataset % 'train',
                                           train_features_dict_path=labeled_features % 'train',
                                           test_dataset_path=labeled_dataset % 'test',
                                           test_features_dict_path=labeled_features % 'test',
                                           word_vector_model_path=args.word_vector_model_path,
                                           train_dataset_extra=labeled_extra % 'train'
                                           if labeled_extra is not None else None,
                                           test_dataset_extra=labeled_extra % 'test'
                                           if labeled_extra is not None else None)

    print('Loading unlabeled dataset', file=sys.stderr)
    unlabeled_dataset = UnlabeledCorpusDataset(dataset_path=unlabeled_dataset,
                                               features_dict_path=unlabeled_features,
                                               word_vector_model=labeled_datasets.train_dataset.word_vector_model,
                                               dataset_extra=unlabeled_dataset_extra)

    prediction_results = []
    certainty_progression = []
    features_progression = []

    print('Running experiments per lemma', file=sys.stderr)
    for lemma, data, target, features in \
            tqdm(labeled_datasets.train_dataset.traverse_dataset_by_lemma(return_features=True)):
        try:
            tf.reset_default_graph()
            with tf.Session() as sess:
                semisupervised = SemiSupervisedWrapper(
                    labeled_train_data=data, labeled_train_target=target,
                    labeled_test_data=labeled_datasets.test_dataset.data(lemma),
                    labeled_test_target=labeled_datasets.test_dataset.target(lemma),
                    unlabeled_data=unlabeled_dataset.data(lemma, limit=args.unlabeled_data_limit),
                    labeled_features=features, min_count=args.min_count, validation_ratio=args.validation_ratio,
                    acceptance_threshold=args.acceptance_threshold, candidates_selection=args.candidates_selection,
                    unlabeled_features=unlabeled_dataset.features_dictionaries(lemma, limit=args.unlabeled_data_limit),
                    candidates_limit=args.candidates_limit, error_sigma=args.error_sigma, random_seed=args.random_seed)

                semisupervised.run(_CLASSIFIERS[args.classifier], config)
                pr, cp, fp = semisupervised.get_results()
                pr.insert(0, 'lemma', lemma)
                cp.insert(0, 'lemma', lemma)
                fp.insert(0, 'lemma', lemma)
                prediction_results.append(pr)
                certainty_progression.append(cp)
                features_progression.append(fp)

                # Save the bootstrapped data
                bootstrapped_indices, bootstrapped_target = semisupervised.bootstrapped()
                unlabeled_instances = [':'.join(ui) for idx, ui in
                                       enumerate(unlabeled_dataset.instances_id(lemma, limit=args.unlabeled_data_limit))
                                       if idx in set(bootstrapped_indices)]

                pd.DataFrame({'instance': unlabeled_instances, 'predicted_target': bootstrapped_target})\
                    .to_csv('%s_unlabeled_dataset_predictions.csv' % args.base_results_path, index=False)
        except ValueError:
            tqdm.write('The lemma %s doesn\'t have enough senses with at least %d occurrences' % (lemma, 2),
                       file=sys.stderr)
            continue

        print('Saving results', file=sys.stderr)
        pd.concat(prediction_results, ignore_index=True)\
            .to_csv('%s_prediction_results.csv' % args.base_results_path, index=False, float_format='%.2e')
        pd.concat(certainty_progression, ignore_index=True)\
            .to_csv('%s_certainty_progression.csv' % args.base_results_path, index=False, float_format='%.2e')
        pd.concat(features_progression, ignore_index=True)\
            .to_csv('%s_features_progression' % args.base_results_path, index=False, float_format='%.2e')
