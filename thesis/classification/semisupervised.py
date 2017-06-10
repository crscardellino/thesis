# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import numpy as np
import pandas as pd
import scipy.sparse as sps
import sys

from itertools import compress
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import StratifiedKFold, KFold
from thesis.classification import BaselineClassifier
from thesis.constants import RANDOM_SEED
from thesis.dataset.utils import filter_minimum, validation_split
from tqdm import tqdm


def _feature_transformer(feature):
    if isinstance(feature[1], str):
        return '='.join(feature), 1
    else:
        return feature


class SemiSupervisedWrapper(object):
    def __init__(self, labeled_train_data, labeled_train_target, labeled_test_data, labeled_test_target,
                 unlabeled_data, labeled_features, unlabeled_features, min_count=2, validation_ratio=0.1,
                 lemma='', acceptance_threshold=0.8, candidates_selection='max', candidates_limit=0, error_sigma=0.1,
                 folds=0, random_seed=RANDOM_SEED, acceptance_alpha=0.05, error_alpha=0.05):
        filtered_values = filter_minimum(target=labeled_train_target[:], min_count=min_count)

        if folds > 0:
            self._labeled_train_data = labeled_train_data[filtered_values]
            self._labeled_train_target = labeled_train_target[filtered_values]
            self._labeled_validation_data = np.array([])
            self._labeled_validation_target = np.array([])
            self._labeled_features = [labeled_features[idx] for idx in filtered_values]
        else:
            train_index, validation_index = validation_split(target=labeled_train_target[filtered_values],
                                                             validation_ratio=validation_ratio, random_seed=random_seed)
            self._labeled_train_data = labeled_train_data[filtered_values][train_index]
            self._labeled_train_target = labeled_train_target[filtered_values][train_index]
            self._labeled_validation_data = labeled_train_data[filtered_values][validation_index]
            self._labeled_validation_target = labeled_train_target[filtered_values][validation_index]
            self._labeled_features = [labeled_features[idx] for idx in filtered_values[train_index]]

        self._labeled_test_data = labeled_test_data
        self._labeled_test_target = labeled_test_target
        self._unlabeled_data = unlabeled_data
        self._unlabeled_features = unlabeled_features

        self._lemma = lemma
        self._classes = np.unique(self._labeled_train_target)
        self._bootstrapped_indices = []
        self._bootstrapped_targets = []
        self._model = None

        self._prediction_results = []
        self._error_progression = []
        self._error_sigma = error_sigma
        self._error_alpha = error_alpha
        self._features_progression = []
        self._certainty_progression = []
        self._cross_validation_results = []

        self._folds = folds
        self._acceptance_threshold = acceptance_threshold
        self._acceptance_alpha = acceptance_alpha
        self._candidates_selection = candidates_selection
        self._candidates_limit = candidates_limit

    @property
    def classes(self):
        return self._classes

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
            over_threshold = np.where(max_probabilities[candidates].round(2) >= self._acceptance_threshold)[0]
            candidates = candidates[over_threshold]

        return candidates

    def _get_target_candidates(self, prediction_probabilities=None, candidates=None):
        raise NotImplementedError

    def _add_results(self, corpus_split, iteration):
        if corpus_split == 'validation' and self._folds > 0:
            # If we evaluate using folds, validation results doesn't matter
            return

        # Get the labeled data/target
        data = getattr(self, '_labeled_%s_data' % corpus_split)
        target = getattr(self, '_labeled_%s_target' % corpus_split)

        # For train corpus we need to append the bootstrapped data and targets
        if corpus_split == 'train':
            data = (data, self._unlabeled_data[self._bootstrapped_indices])
            data = sps.vstack(data) if sps.issparse(self._labeled_train_data) else np.vstack(data)
            target = np.concatenate((target, self._bootstrapped_targets))

            # Add the features of the new data to the progression
            unlabeled_features = [self._unlabeled_features[idx] for idx in self._bootstrapped_indices]
            features = self._labeled_features + unlabeled_features

            for tgt, feats in zip(target, features):
                feats = [_feature_transformer(f) for f in sorted(feats.items())]
                fdf = pd.DataFrame(feats, columns=['feature', 'count'])
                fdf.insert(0, 'target', np.int(tgt))
                fdf.insert(0, 'iteration', iteration)

                self._features_progression.append(fdf)

        # Calculate cross entropy error (perhaps better with the algorithm by itself)
        # and update the results of the iteration giving the predictions
        results = pd.DataFrame({'true': target.astype(np.int32),
                                'prediction': self._model.predict(data).astype(np.int32)},
                               columns=['true', 'prediction'])
        results.insert(0, 'iteration', iteration)
        results.insert(0, 'corpus_split', corpus_split)

        # Add the results to the corresponding corpus split results
        self._prediction_results.append(results)

    def _validate(self, model_class, model_config, train_data, train_target, iteration):
        if self._folds > 0:
            validation_errors = []
            cross_validation = []
            cv = StratifiedKFold(self._folds)
            try:
                _ = next(cv.split(train_data, train_target))
            except ValueError:
                cv = KFold(self._folds)

            for fold_no, (train_indices, test_indices) in enumerate(cv.split(train_data, train_target), start=1):
                cv_train_data = train_data[train_indices]
                cv_test_data = train_data[test_indices]
                cv_train_target = train_target[train_indices]
                cv_test_target = train_target[test_indices]

                try:
                    model = model_class(**model_config)
                    model.fit(cv_train_data, cv_train_target)
                except ValueError:
                    if np.unique(cv_train_target).shape[0] > 1:
                        raise
                    else:
                        model = BaselineClassifier()
                        model.fit(cv_train_data, cv_train_target)

                cv_train_results = pd.DataFrame(
                    {'true': cv_train_target.astype(np.int32),
                     'prediction': model.predict(cv_train_data).astype(np.int32)},
                    columns=['true', 'prediction']
                )
                cv_train_results.insert(0, 'fold', fold_no)
                cv_train_results.insert(0, 'corpus_split', 'train')
                cv_train_results.insert(0, 'iteration', iteration)

                cv_test_results = pd.DataFrame(
                    {'true': cv_test_target.astype(np.int32),
                     'prediction': model.predict(cv_test_data).astype(np.int32)},
                    columns=['true', 'prediction']
                )
                cv_test_results.insert(0, 'fold', fold_no)
                cv_test_results.insert(0, 'corpus_split', 'test')
                cv_test_results.insert(0, 'iteration', iteration)

                validation_errors.append(zero_one_loss(cv_test_results.true, cv_test_results.prediction))
                cross_validation.append(pd.concat([cv_train_results, cv_test_results], ignore_index=True))

            return validation_errors, cross_validation, np.mean(validation_errors), None
        else:
            new_model = model_class(**model_config)
            new_model.fit(train_data, train_target)

            validation_error = zero_one_loss(
                self._labeled_validation_target,
                new_model.predict(self._labeled_validation_data)
            )

            return None, None, validation_error, new_model

    @property
    def error_sigma(self):
        return self._error_sigma

    def bootstrapped(self):
        return self._bootstrapped_indices, self._bootstrapped_targets

    def get_results(self):
        prediction_results = pd.concat(self._prediction_results, ignore_index=True)
        certainty_progression = pd.concat(self._certainty_progression, ignore_index=True)
        features_progression = pd.concat(self._features_progression, ignore_index=True)

        cross_validation_results = pd.concat(self._cross_validation_results, ignore_index=True)\
            if self._folds > 0 else None

        return prediction_results, certainty_progression, features_progression, cross_validation_results

    def run(self, model_class, model_config):
        self._model = model_class(**model_config)
        self._model.fit(self._labeled_train_data, self._labeled_train_target)

        for corpus_split in ('train', 'validation', 'test'):
            self._add_results(corpus_split, 'initial')

        # Check validation error for initial model
        validation_errors, cross_validation, validation_error, _ = self._validate(
            model_class, model_config, self._labeled_train_data, self._labeled_train_target, 'initial'
        )

        self._error_progression.append(validation_error)
        if self._folds > 0:
            self._cross_validation_results.extend(cross_validation)

        iteration = 0
        bootstrap_mask = np.ones(self._unlabeled_data.shape[0], dtype=np.bool)
        unlabeled_dataset_index = np.arange(self._unlabeled_data.shape[0], dtype=np.int32)

        while len(self._bootstrapped_indices) < self._unlabeled_data.shape[0]:
            bootstrap_mask[self._bootstrapped_indices] = False
            masked_unlabeled_data = self._unlabeled_data[bootstrap_mask]
            prediction_probabilities = self._model.predict_proba(masked_unlabeled_data)

            candidates = self._get_candidates(prediction_probabilities)

            if len(candidates) == 0:  # Can't get more candidates
                if iteration == 0:
                    # Check there is at least 1 iteration running. If not, adapt the acceptance threshold
                    self._acceptance_threshold -= self._acceptance_alpha
                    if self._acceptance_threshold > 0.5:  # Otherwise is too low
                        continue
                    else:
                        break
                else:  # There was at least one iteration.
                    tqdm.write('Lemma %s - Max predicted probability %.2f - Acceptance threshold: %.2f'
                               % (self._lemma, prediction_probabilities.max(), self._acceptance_threshold),
                               file=sys.stderr)
                    break

            data_candidates = masked_unlabeled_data[candidates]
            target_candidates = self._get_target_candidates(prediction_probabilities, candidates)

            train_data = (self._labeled_train_data, self._unlabeled_data[self._bootstrapped_indices], data_candidates)
            train_data = sps.vstack(train_data) if sps.issparse(self._labeled_train_data) else np.vstack(train_data)
            train_target = np.concatenate((self._labeled_train_target, self._bootstrapped_targets, target_candidates))

            assert train_data.shape[0] == train_target.shape[0],\
                'The train data and target have different shapes: %d != %d' % (train_data.shape[0],
                                                                               train_target.shape[0])

            # Train the new model and check validation
            validation_errors, cross_validation, validation_error, new_model = self._validate(
                model_class, model_config, train_data, train_target, iteration
            )

            min_progression_error = min(self._error_progression)

            if self._error_sigma > 0 and validation_error > min_progression_error + self._error_sigma:
                if iteration == 0:
                    # Check there is at least 1 iteration running. If not, adapt the error_sigma
                    self._error_sigma += self._error_alpha
                    if self._error_sigma < 0.5: # So the error is not so high
                        continue
                    else:
                        break
                else:  # There was at least one iteration.
                    tqdm.write('Lemma %s - Validation error: %.2f - Progression min error: %.2f'
                               % (self._lemma, validation_error, min_progression_error), file=sys.stderr)
                    break

            if self._folds > 0:
                self._cross_validation_results.extend(cross_validation)
                self._model = model_class(**model_config)
                self._model.fit(train_data, train_target)
            else:
                self._model = new_model

            self._bootstrapped_indices.extend(unlabeled_dataset_index[bootstrap_mask][candidates])
            self._bootstrapped_targets.extend(target_candidates)
            self._error_progression.append(validation_error)
            iteration += 1

            # Add the certainty of the predicted classes of the unseen examples to the certainty progression results
            certainty_df = pd.DataFrame(prediction_probabilities.max(axis=1), columns=['certainty'])
            certainty_df.insert(0, 'iteration', iteration)
            self._certainty_progression.append(certainty_df)

            for corpus_split in ('train', 'validation'):
                self._add_results(corpus_split, iteration)

        if len(self._bootstrapped_indices) >= self._unlabeled_data.shape[0]:
            tqdm.write('Lemma %s - Run all iterations' % self._lemma, file=sys.stderr)

        for corpus_split in ('train', 'test'):
            self._add_results(corpus_split, 'final')

        return iteration


class SelfLearningWrapper(SemiSupervisedWrapper):
    def _get_target_candidates(self, prediction_probabilities=None, candidates=None):
        return self._classes[prediction_probabilities[candidates].argmax(axis=1)]


class ActiveLearningWrapper(SemiSupervisedWrapper):
    def __init__(self, **kwargs):
        self._unlabeled_target = kwargs.pop('unlabeled_target', np.array([]))
        self._unlabeled_sentences = kwargs.pop('unlabeled_sentences', None)
        self._train_classes = kwargs.pop('train_classes', None)
        full_senses_dict = kwargs.pop('full_senses_dict', None)

        super(ActiveLearningWrapper, self).__init__(**kwargs)

        self._senses = sorted(full_senses_dict[self._lemma].items()) if full_senses_dict else None

    def _get_target_candidates(self, prediction_probabilities=None, candidates=None):
        bootstrap_mask = np.ones(self._unlabeled_data.shape[0], dtype=np.bool)
        bootstrap_mask[self._bootstrapped_indices] = False

        if self._unlabeled_target.shape[0] > 0:
            return self._unlabeled_target[bootstrap_mask][candidates]
        else:
            ul_sentences = list(compress(self._unlabeled_sentences, bootstrap_mask))
            ul_sentences = [ul_sentences[idx] for idx in candidates]
            labeled_targets = []

            for sentence in ul_sentences:
                print('*' * 50 + '\n%s' % sentence, file=sys.stderr)
                print('*' * 50 + '\nSelect the sense for the previous sentence:', file=sys.stderr)
                for idx, (sense, description) in enumerate(self._senses):
                    print('%d ) %s: %s' % (idx, sense, description), file=sys.stderr)
                sense = input('Sense: ')
                while not sense.isdigit() or not (0 <= int(sense) < len(self._senses)):
                    sense = input('Sense: ')
                sense = int(sense)

                print(file=sys.stderr)

                if self._senses[sense][0] not in self._train_classes:
                    self._train_classes[self._senses[sense][0]] = len(self._train_classes)

                labeled_targets.append(self._train_classes[self._senses[sense][0]])

            return labeled_targets

    def get_senses(self):
        senses_description = dict(self._senses)
        senses = []
        for sense in self._train_classes:
            senses.append((self._train_classes[sense], sense,
                           senses_description[sense] if sense in senses_description else 'NA'))
        return pd.DataFrame(senses, columns=['id', 'sense', 'description'])
