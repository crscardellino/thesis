# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1, l2, l1l2
from keras.utils.np_utils import to_categorical
from scipy.sparse import issparse


class KerasMultilayerPerceptron(object):
    _MAX_BATCH_SIZE = 3000

    def __init__(self, layers=list(), layers_activation='tanh', classification_layer_activation='softmax',
                 layers_initialization='uniform', dropout_layers=None, optimizer='adam',
                 loss='categorical_crossentropy', epochs=25, batch_size=50, metrics=None, l1=0., l2=1e-3,
                 verbosity=0):
        self._layers = layers
        self._layers_activation = layers_activation
        self._classification_layer_activation = classification_layer_activation
        self._layers_initialization = layers_initialization
        self._dropout_layers = dropout_layers
        self._optimizer = optimizer
        self._loss = loss
        self._epochs = epochs
        self._batch_size = batch_size
        self._metrics = metrics
        self._l1 = l1
        self._l2 = l2
        self._verbosity = verbosity
        self._model = None
        self._classes = None

    def _build_network(self, input_size, output_size):
        self._model = Sequential()

        for lidx, layer_dim in enumerate(self._layers + [output_size]):
            layer_config = {
                'output_dim': layer_dim,
                'input_dim': None if self._model.layers else input_size,
                'init': self._layers_initialization,
            }

            if lidx < len(self._layers):
                layer_config['activation'] = self._layers_activation
            else:
                layer_config['activation'] = self._classification_layer_activation

            if self._l1 > 0 and self._l2 > 0:
                layer_config['W_regularizer'] = l1l2(self._l1, self._l2)
                layer_config['b_regularizer'] = l1l2(self._l1, self._l2)
            elif self._l1 > 0:
                layer_config['W_regularizer'] = l1(self._l1)
                layer_config['b_regularizer'] = l1(self._l1)
            elif self._l2 > 0:
                layer_config['W_regularizer'] = l2(self._l2)
                layer_config['b_regularizer'] = l2(self._l2)

            self._model.add(Dense(**layer_config))

            if self._dropout_layers:
                self._model.add(Dropout(self._dropout_layers.pop(0)))

        self._model.compile(loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)

    def fit(self, x, y):
        self._classes = np.unique(y)
        self._build_network(x.shape[1], len(self._classes))

        if self._epochs < 0 and self._batch_size < 0:
            self._batch_size = min(x.shape[0], self._MAX_BATCH_SIZE)
            self._epochs = np.int(np.ceil(x.shape[0] / self._batch_size))
        elif self._epochs < 0:
            self._epochs = np.int(np.ceil(x.shape[0] / self._batch_size))
        elif self._batch_size < 0:
            self._batch_size = np.int(np.ceil(x.shape[0] / self._epochs))

        x = x.toarray() if issparse(x) else x

        return self._model.fit(x, to_categorical(np.searchsorted(self._classes, y)),
                               nb_epoch=self._epochs, batch_size=self._batch_size, verbose=self._verbosity)

    def predict(self, x):
        """
        Predicts the model classes (defined in self._classes) for the input data
        """
        assert self._model is not None, "The model needs to be trained"

        x = x.toarray() if issparse(x) else x
        predicted_classes = self._model.predict_classes(x, batch_size=self._batch_size, verbose=self._verbosity)

        return self._classes[predicted_classes]

    def predict_proba(self, x):
        """
        Predicts the model class probability for the input data
        """
        assert self._model is not None, "The model needs to be trained"

        x = x.toarray() if issparse(x) else x
        return self._model.predict(x, verbose=self._verbosity)
