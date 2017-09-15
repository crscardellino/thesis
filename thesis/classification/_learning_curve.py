# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf

from keras import backend as keras_backend
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from thesis.utils import cumulative_index_split


def learning_curve_training(estimator, data, target, estimator_config=None, folds=3, splits=3, min_count=2,
                            test_size=0.2, random_seed=1234):
    for split_no, indices in enumerate(cumulative_index_split(target, splits, min_count), start=1):
        train_data = data[indices]
        train_target = target[indices]

        if folds > 1:
            cv = StratifiedKFold(folds)
            try:
                _ = next(cv.split(train_data, train_target))
            except ValueError:
                cv = KFold(folds)
            for fold_no, (train_indices, test_indices) in enumerate(cv.split(train_data, train_target), start=1):
                tf.reset_default_graph()
                with tf.Session() as sess:
                    keras_backend.set_session(sess)

                    fold_train_data = train_data[train_indices]
                    fold_test_data = train_data[test_indices]
                    fold_train_target = train_target[train_indices]
                    fold_test_target = train_target[test_indices]

                    model = estimator(**(estimator_config or dict()))
                    model.fit(fold_train_data, fold_train_target)

                    fold_train_results = pd.DataFrame(
                        {'true': fold_train_target.astype(np.int32),
                         'prediction': model.predict(fold_train_data).astype(np.int32)},
                        columns=['true', 'prediction']
                    )
                    fold_train_results.insert(0, 'fold', fold_no)
                    fold_train_results.insert(0, 'corpus_split', 'train')

                    fold_test_results = pd.DataFrame(
                        {'true': fold_test_target.astype(np.int32),
                         'prediction': model.predict(fold_test_data).astype(np.int32)},
                        columns=['true', 'prediction']
                    )
                    fold_test_results.insert(0, 'fold', fold_no)
                    fold_test_results.insert(0, 'corpus_split', 'test')

                    learning_curve_results = pd.concat([fold_train_results, fold_test_results], ignore_index=True)
                    learning_curve_results.insert(0, 'percent_size', split_no / np.float(splits))
                    learning_curve_results.insert(0, 'split', split_no)

                    yield learning_curve_results
        else:
            tf.reset_default_graph()
            tf.set_random_seed(random_seed)
            with tf.Session() as sess:
                keras_backend.set_session(sess)

                train_indices, test_indices = train_test_split(np.arange(train_data.shape[0]), test_size=test_size)
                while np.unique(train_target[train_indices]).shape[0] > 1:
                    train_indices, test_indices = train_test_split(np.arange(train_data.shape[0]), test_size=test_size)

                fold_train_data = train_data[train_indices]
                fold_test_data = train_data[test_indices]
                fold_train_target = train_target[train_indices]
                fold_test_target = train_target[test_indices]

                model = estimator(**(estimator_config or dict()))
                model.fit(fold_train_data, fold_train_target)

                fold_train_results = pd.DataFrame(
                    {'true': fold_train_target.astype(np.int32),
                     'prediction': model.predict(fold_train_data).astype(np.int32)},
                    columns=['true', 'prediction']
                )
                fold_train_results.insert(0, 'fold', 1)
                fold_train_results.insert(0, 'corpus_split', 'train')

                fold_test_results = pd.DataFrame(
                    {'true': fold_test_target.astype(np.int32),
                     'prediction': model.predict(fold_test_data).astype(np.int32)},
                    columns=['true', 'prediction']
                )
                fold_test_results.insert(0, 'fold', 1)
                fold_test_results.insert(0, 'corpus_split', 'test')

                learning_curve_results = pd.concat([fold_train_results, fold_test_results], ignore_index=True)
                learning_curve_results.insert(0, 'num_examples', fold_train_data.shape[0])
                learning_curve_results.insert(0, 'percent_size', split_no / np.float(splits))
                learning_curve_results.insert(0, 'split', split_no)

                yield learning_curve_results
