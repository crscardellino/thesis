# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import scipy.sparse as sps
import sys

from sklearn.metrics import log_loss
from thesis.dataset import SenseCorpusDatasets, UnlabeledCorpusDataset
from thesis.dataset.utils import filter_minimum, validation_split
from thesis.utils import RANDOM_SEED
from tqdm import tqdm


class SemiSupervisedWrapper(object):
    def __init__(self, labeled_train_data, labeled_train_target, labeled_test_data, labeled_test_target,
                 unlabeled_data, min_count=2, validation_ratio=0.1, acceptance_threshold=0.8,
                 candidates_selection='max', candidates_limit=0, error_sigma=0.001, random_seed=RANDOM_SEED):
        labeled_train_data, labeled_train_target =\
            filter_minimum(labeled_train_data, labeled_train_target, min_count)
        labeled_train_data, labeled_validation_data, labeled_train_target, labeled_validation_target =\
            validation_split(data=labeled_train_target, target=labeled_train_target,
                             validation_ratio=validation_ratio, random_seed=random_seed)

        self._labeled_train_data = labeled_train_data
        self._labeled_train_target = labeled_train_target
        self._labeled_validation_data = labeled_validation_data
        self._labeled_validation_target = labeled_validation_target
        self._labeled_test_data = labeled_test_data
        self._labeled_test_target = labeled_test_target
        self._unlabeled_data = unlabeled_data

        self._labels = np.unique(self._labeled_train_target)
        self._bootstrapped_indices = []
        self._bootstrapped_targets = []
        self._model = None

        self._train_results = []
        self._validation_results = []
        self._test_results = []
        self._error_progression = []
        self._error_sigma = error_sigma

        self._acceptance_threshold = acceptance_threshold
        self._candidates_selection = candidates_selection
        self._candidates_limit = candidates_limit

    def _get_candidates(self, prediction_probabilities):
        # Get the max probabilities per target
        max_probabilities = prediction_probabilities.max(axis=1)

        # Sort the candidate probabilities according to the selection method
        if self._candidates_selection == 'min':
            candidate_indices = np.argsort(max_probabilities)
        elif self._candidates_selection == 'max':
            candidate_indices = np.argsort(max_probabilities)[::-1]
        elif self._candidates_selection == 'random':
            candidate_indices = np.random.permutation(max_probabilities.shape[0])
        else:
            raise ValueError('Not a valida candidate selection method: %s' % self._candidates_selection)

        # Select the candidates, limiting them in case of an existing limit
        candidates = max_probabilities[candidate_indices]
        if self._candidates_limit > 0:
            candidates = candidates[:self._candidates_limit]

        # If there is an acceptance threshold filter out candidates that doesn't comply it
        if self._acceptance_threshold > 0:
            over_threshold = np.where(max_probabilities[candidates] >= self._acceptance_threshold)[0]
            candidates = candidates[over_threshold]

        return candidates

    def _add_results(self, corpus_split, iteration):
        # FIXME: There are metrics left out, mostly related to features and so
        # Get the labeled data/target unless given by the parameters
        # (e.g. when using initial train data + bootstrapped data)
        data = getattr(self, '_labeled_%s_data' % corpus_split)
        target = getattr(self, '_labeled_%s_target' % corpus_split)

        # For train corpus we need to append the bootstrapped data and targets
        if corpus_split == 'train':
            data = (data, self._unlabeled_data[self._bootstrapped_indices])
            data = sps.vstack(data) if sps.issparse(self._labeled_train_data) else np.vstack(data)
            target = np.concatenate((target, self._bootstrapped_targets))

        # Calculate cross entropy error (perhaps better with the algorithm by itself)
        # and update the results of the iteration giving the predictions
        error = log_loss(target, self._model.predict_proba(data), labels=self._labels)
        results = pd.DataFrame(np.vstack([target, self._model.predict(data)]).T,
                               columns=['true', 'prediction'])
        results.insert(0, 'error', error)
        results.insert(0, 'iteration', iteration)
        results.insert(0, 'corpus_split', corpus_split)

        # For the initial iteration of the validation corpus, we append the error progression
        if corpus_split == 'validation' and iteration == 'initial':
            self._error_progression.append(error)

        # Add the results to the corresponding corpus split results
        getattr(self, '_%s_results' % corpus_split).append(results)

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
            for corpus_split in ('train', 'validation'):
                self._add_results(corpus_split, iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labeled_dataset',
                        type=str)
    parser.add_argument('unlabeled_dataset',
                        type=str)

    args = parser.parse_args()

    labeled_datasets = SenseCorpusDatasets(os.path.join(args.labeled_dataset, 'train_dataset.npz'),
                                           os.path.join(args.labeled_dataset, 'test_dataset.npz'),
                                           os.path.join(args.labeled_dataset, 'train_features.p'),
                                           os.path.join(args.labeled_dataset, 'test_features.p'))
    unlabeled_dataset = UnlabeledCorpusDataset(os.path.join(args.unlabeled_dataset, 'dataset.npz'),
                                               os.path.join(args.unlabeled_dataset, 'features.p'))

    for lemma, data, target in tqdm(labeled_datasets.train_dataset.traverse_dataset_by_lemma()):
        # FIXME
        try:
            semisupervised = SemiSupervisedWrapper(data, target,
                                                   labeled_datasets.test_dataset.data(lemma),
                                                   labeled_datasets.test_dataset.target(lemma),
                                                   unlabeled_dataset.data(lemma), 2, 0.1, 1234)
        except ValueError:
            print('The lemma %s doesn\'t have enough senses with at least %d occurrences' % (lemma, 2),
                  file=sys.stderr)
            continue
