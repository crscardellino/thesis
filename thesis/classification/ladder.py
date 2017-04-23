# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

from keras.utils.np_utils import to_categorical
from thesis.dataset import SenseCorpusDatasets, UnlabeledCorpusDataset
from thesis.dataset.utils import filter_minimum, validation_split, NotEnoughSensesError
from thesis.utils import RANDOM_SEED
from tqdm import tqdm, trange


class LadderNetworksExperiment(object):
    def __init__(self, labeled_train_data, labeled_train_target, labeled_test_data, labeled_test_target,
                 unlabeled_data, layers, denoising_cost, min_count=2, validation_ratio=0.1, epochs=25,
                 noise_std=0.3, learning_rate=0.01, evaluation_amount=100, random_seed=RANDOM_SEED):
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

        self._classes = np.unique(self._labeled_train_target)
        self._prediction_results = []

        self._input_size = self._labeled_train_data.shape[1]
        self._output_size = self._classes.shape[0]

        self._layers = layers
        self._layers.insert(0, self._input_size)
        self._layers.append(self._output_size)
        self._L = len(self._layers) - 1  # size of layers ignoring input layer

        self._labeled_num_examples = self._labeled_train_data.shape[0]
        self._unlabeled_num_examples = self._unlabeled_data.shape[0]
        self._num_examples = self._labeled_num_examples + self._unlabeled_num_examples
        self._num_epochs = epochs
        self._epochs_completed = 0
        self._batch_size = self._labeled_num_examples
        self._num_iter = (self._num_examples / self._batch_size) * self._num_epochs

        # keep track of epochs
        self._labeled_permutation = np.arange(self._labeled_num_examples)
        self._unlabeled_permutation = np.arange(self._unlabeled_num_examples)
        self._unlabeled_index_in_epoch = 0

        # ladder networks hyperparameters
        self._noise_std = noise_std  # scaling factor for noise used in corrupted encoder
        self._denoising_cost = denoising_cost  # hyperparameters that denote the importance of each layer

        # functions to join and split labeled and unlabeled corpus
        self._join = lambda l, u: tf.concat(0, [l, u])
        self._labeled = lambda i: tf.slice(i, [0, 0], [self._batch_size, -1]) if i is not None else i
        self._unlabeled = lambda i: tf.slice(i, [self._batch_size, 0], [-1, -1]) if i is not None else i
        self._split_lu = lambda i: (self._labeled(i), self._unlabeled(i))

        self._build_network()

        self._y_c, self._corrupted_encoder = self._encoder(self._noise_std)

        _, self._clean_encoder = self._encoder(0.0)
        # the y function is ignored as is only helpful for evaluation and classification

        # define the y function as the classification function
        self._y = self._mlp(self._inputs)

        # calculate total unsupervised cost by adding the denoising cost of all layers
        self._uloss = tf.add_n(self._decoder())

        self._y_N = self._labeled(self._y_c)
        self._lloss = -tf.reduce_mean(tf.reduce_sum(self._outputs * tf.log(self._y_N), 1))

        # total cost (loss function)
        self._loss = self._lloss + self._uloss

        # y_true and y_pred used to get the metrics
        self._y_true = tf.argmax(self._outputs, 1)
        self._y_pred = tf.argmax(self._y, 1)

        # evaluation_sentences
        self._evaluation_amount = evaluation_amount
        self._evaluation_sentences_indices = []
        self._evaluation_sentences_indices_set = set()
        self._evaluation_sentences_target = []

        # train_step for the weight parameters, optimized with Adam
        self._learning_rate = tf.Variable(learning_rate, trainable=False)
        self._train_step = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)

        # add the updates of batch normalization statistics to train_step
        bn_updates = tf.group(*self._bn_assigns)
        with tf.control_dependencies([self._train_step]):
            self._train_step = tf.group(bn_updates)

    @property
    def evaluation_sentences(self):
        return self._evaluation_sentences_indices, self._evaluation_sentences_target

    def get_results(self):
        return pd.concat(self._prediction_results, ignore_index=True)

    def _add_result(self, sess, corpus_split, feed_dict, epoch):
        serror, uerror, y_true, y_pred = sess.run(
            [self._lloss, self._uloss, self._y_true, self._y_pred], feed_dict=feed_dict
        )
        # Calculate cross entropy error (perhaps better with the algorithm by itself)
        # and update the results of the iteration giving the predictions
        results = pd.DataFrame({'true': y_true.astype(np.int32),
                                'prediction': y_pred.astype(np.int32)})
        results.insert(0, 'supervised_error', serror)
        results.insert(0, 'unsupervised_error', uerror)
        results.insert(0, 'full_error', serror + uerror)
        results.insert(0, 'epoch', epoch)
        results.insert(0, 'corpus_split', corpus_split)

        # Add the results to the corresponding corpus split results
        self._prediction_results.append(results)

    def _build_network(self):
        # input of the network (will be use to place the examples for training and classification)
        self._inputs = tf.placeholder(tf.float32, shape=(None, self._input_size))

        # output of the network (will be use to place the labels of the examples for training and testing)
        self._outputs = tf.placeholder(tf.float32)

        # lambda functions to create the biases and weight (matrices) variables of the network
        bi = lambda inits, size, name: tf.Variable(inits * tf.ones([size]), name=name)
        # a bias has an initialization value (generally either one or zero), a size (lenght of the vector)
        # and a name

        wi = lambda shape, name: tf.Variable(tf.random_normal(shape, name=name)) / np.sqrt(shape[0])
        # a weight has a shape of the matrix (inputs from previous layers outputs of the next layer)
        # and is initialized as a random normal divided by the lenght of the previous layer.

        shapes = list(zip(self._layers[:-1], self._layers[1:]))
        # the shape of each of the linear layers, is needed to build the structure of the network

        # define the weights, randomly initialized at first, for the encoder and the decoder.
        # also define the biases for shift and scale the normalized values of a batch
        self._weights = dict(
            W=[wi(s, 'W') for s in shapes],  # encoder weights
            V=[wi(s[::-1], 'V') for s in shapes],  # decoder weights
            beta=[bi(0.0, self._layers[l+1], 'beta') for l in range(self._L)],
            # batch normalization parameter to shift the normalized value
            gamma=[bi(1.0, self._layers[l+1], 'gamma') for l in range(self._L)]
            # batch normalization parameter to scale the normalized value
        )

        # calculates the moving averages of mean and variance, needed for batch
        # normalization of the decoder step at each layer
        self._ewma = tf.train.ExponentialMovingAverage(decay=0.99)
        # stores the updates to be made to average mean and variance
        self._bn_assigns = []

        # average mean and variance of all layers
        self._running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in self._layers[1:]]
        self._running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in self._layers[1:]]

    def _update_batch_normalization(self, batch, l):
        # batch normalize + update average mean and variance of layer l
        mean, var = tf.nn.moments(batch, axes=[0])
        assign_mean = self._running_mean[l-1].assign(mean)
        assign_var = self._running_var[l-1].assign(var)
        self._bn_assigns.append(self._ewma.apply([self._running_mean[l-1], self._running_var[l-1]]))
        with tf.control_dependencies([assign_mean, assign_var]):
            return (batch - mean) / tf.sqrt(var + 1e-10)

    def _decoder(self):
        z_est = {}
        # stores the denoising cost of all layers
        d_cost = []

        for l in range(self._L, -1, -1):
            z, z_c = self._clean_encoder['unlabeled']['z'][l], self._corrupted_encoder['unlabeled']['z'][l]

            m = self._clean_encoder['unlabeled']['m'].get(l, 0)
            v = self._clean_encoder['unlabeled']['v'].get(l, 1-1e-10)

            if l == self._L:
                u = self._unlabeled(self._y_c)
            else:
                u = tf.matmul(z_est[l+1], self._weights['V'][l])

            u = self._batch_normalization(u)
            z_est[l] = self._combinator_g(z_c, u, self._layers[l])
            z_est_bn = (z_est[l] - m) / v
            # append the cost of this layer to d_cost
            d_cost.append(
                    (tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / self._layers[l]) *
                    self._denoising_cost[l]
            )

        return d_cost

    def _encoder(self, noise_std):
        """
        encoder factory for training.
        """
        # add noise to input
        h = self._inputs + tf.random_normal(tf.shape(self._inputs)) * noise_std

        # dictionary to store the pre-activation, activation, mean and variance for each layer
        layer_data = dict()

        # the data for labeled and unlabeled examples are stored separately
        layer_data['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        layer_data['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}

        # get the data for the input layer, divided in labeled and unlabeled
        layer_data['labeled']['z'][0], layer_data['unlabeled']['z'][0] = self._split_lu(h)
        l = 1
        for l in range(1, self._L+1):
            layer_data['labeled']['h'][l - 1], layer_data['unlabeled']['h'][l - 1] = self._split_lu(h)

            # pre-activation
            z_pre = tf.matmul(h, self._weights['W'][l-1])
            # split labeled and unlabeled examples
            z_pre_l, z_pre_u = self._split_lu(z_pre)

            # batch normalization for labeled and unlabeled examples is performed separately
            m, v = tf.nn.moments(z_pre_u, axes=[0])
            if noise_std > 0:
                # Corrupted encoder
                # batch normalization + noise
                z = self._join(self._batch_normalization(z_pre_l), self._batch_normalization(z_pre_u, m, v))
                z += tf.random_normal(tf.shape(z_pre)) * noise_std
            else:
                # Clean encoder
                # batch normalization + update the average mean and variance
                # using batch mean and variance of labeled examples
                z = self._join(self._update_batch_normalization(z_pre_l, l), self._batch_normalization(z_pre_u, m, v))

            if l == self._L:
                # use softmax activation in output layer
                h = tf.nn.softmax(self._weights['gamma'][l-1] * (z + self._weights['beta'][l-1]))
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + self._weights['beta'][l-1])

            layer_data['labeled']['z'][l], layer_data['unlabeled']['z'][l] = self._split_lu(z)

            # save mean and variance of unlabeled examples for decoding
            layer_data['unlabeled']['m'][l], layer_data['unlabeled']['v'][l] = m, v

        # get the h values for unlabeled and labeled for the last layer
        layer_data['labeled']['h'][l], layer_data['unlabeled']['h'][l] = self._split_lu(h)

        return h, layer_data

    def _mlp(self, inputs):
        h = inputs
        for l in range(1, self._L+1):
            # pre-activation
            z_pre = tf.matmul(h, self._weights['W'][l-1])

            # obtain average mean and variance and use it to normalize the batch
            mean = self._ewma.average(self._running_mean[l-1])
            var = self._ewma.average(self._running_var[l-1])
            z = self._batch_normalization(z_pre, mean, var)

            if l == self._L:
                # use softmax activation in output layer
                h = tf.nn.softmax(self._weights['gamma'][l-1] * (z + self._weights['beta'][l-1]))
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + self._weights['beta'][l-1])
        return h

    @staticmethod
    def _batch_normalization(batch, mean=None, var=None):
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=[0])
        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

    @staticmethod
    def _combinator_g(z_c, u, size):
        """
        combinator function for the lateral z_c and the vertical u value in each
        layer of the decoder, proposed by the original paper
        """
        wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')

        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu
        return z_est

    @property
    def _unlabeled_batch(self):
        start = self._unlabeled_index_in_epoch
        self._unlabeled_index_in_epoch += self._batch_size

        if self._unlabeled_index_in_epoch > self._unlabeled_num_examples:
            self._unlabeled_permutation = np.random.permutation(self._unlabeled_permutation)
            start = 0
            self._unlabeled_index_in_epoch = self._batch_size

        end = self._unlabeled_index_in_epoch

        return self._unlabeled_permutation[start:end]

    def run(self):
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            feed_dicts = {
                'train': {
                    self._inputs: self._labeled_train_data,
                    self._outputs: to_categorical(np.searchsorted(self._classes, self._labeled_train_target))
                },
                'test': {
                    self._inputs: self._labeled_test_data,
                    self._outputs: to_categorical(np.searchsorted(self._classes, self._labeled_test_target))
                },
                'validation': {
                    self._inputs: self._labeled_validation_data,
                    self._outputs: to_categorical(np.searchsorted(self._classes, self._labeled_validation_data))
                }
            }

            for corpus_split in ('train', 'test', 'validation'):
                self._add_result(sess, corpus_split, feed_dicts[corpus_split], 'initial')

            for i in trange(1, self._num_iter + 1):
                data = np.vstack((self._labeled_train_data[self._labeled_permutation],
                                  self._unlabeled_data[self._unlabeled_batch]))
                target = to_categorical(
                    np.searchsorted(self._classes, self._labeled_train_target[self._labeled_permutation]))

                _, error = sess.run([self._train_step, self._loss],
                                    feed_dict={self._inputs: data, self._outputs: target})

                if i % (self._num_iter/self._num_epochs) == 0:
                    self._epochs_completed += 1

                    tqdm.write('Epoch %d - error: %f' % (self._epochs_completed, error), file=sys.stderr)

                    for corpus_split in ('train', 'validation'):
                        self._add_result(sess, corpus_split, feed_dicts[corpus_split], self._epochs_completed)

                    # selecting some random unlabeled instances for classification and manual evaluation
                    perm = np.array([uidx for uidx in np.random.permutation(self._unlabeled_num_examples)
                                     if uidx not in self._evaluation_sentences_indices_set])[:self._evaluation_amount]
                    y_pred = sess.run(self._y_pred, feed_dict={self._inputs: self._unlabeled_data[perm]})

                    self._evaluation_sentences_indices.extend(perm)
                    self._evaluation_sentences_target.extend(y_pred)
                    self._evaluation_sentences_indices_set.update(perm)

            for corpus_split in ('train', 'test', 'validation'):
                self._add_result(sess, corpus_split, feed_dicts[corpus_split], 'final')

        del sess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labeled_dataset_path')
    parser.add_argument('unlabeled_dataset_path')
    parser.add_argument('base_results_path')
    parser.add_argument('--word_vector_model_path', default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=list())
    parser.add_argument('--denoising_cost', type=float, nargs='+', default=list())
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--unlabeled_data_limit', type=int, default=1000)
    parser.add_argument('--evaluation_amount', type=int, default=100)
    parser.add_argument('--min_count', type=int, default=2)
    parser.add_argument('--noise_std', type=float, default=0.3)
    parser.add_argument('--validation_ratio', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--random_seed', type=int, default=1234)

    args = parser.parse_args()

    args.layers = [args.layers] if not isinstance(args.layers, list) else args.layers
    args.denoising_cost = [args.denoising_cost] if not isinstance(args.denoising_cost, list) else args.denoising_cost

    if len(args.layers) + 2 != len(args.denoising_cost) or len(args.layers) == 0:
        raise ValueError('Not valid layers or denoising cost')

    labeled_datasets_path = os.path.join(args.labeled_dataset_path, '%s_dataset.npz')
    unlabeled_dataset_path = os.path.join(args.unlabeled_dataset_path, 'dataset.npz')

    print('Loading labeled dataset', file=sys.stderr)
    labeled_datasets = SenseCorpusDatasets(train_dataset_path=labeled_datasets_path % 'train',
                                           test_dataset_path=labeled_datasets_path % 'test',
                                           word_vector_model_path=args.word_vector_model_path)

    print('Loading unlabeled dataset', file=sys.stderr)
    unlabeled_dataset = UnlabeledCorpusDataset(dataset_path=unlabeled_dataset_path,
                                               word_vector_model=labeled_datasets.train_dataset.word_vector_model)

    prediction_results = []
    evaluation_instances = []
    evaluation_targets = []

    print('Running experiments per lemma', file=sys.stderr)
    for lemma, data, target, features in \
            tqdm(labeled_datasets.train_dataset.traverse_dataset_by_lemma(return_features=True),
                 total=labeled_datasets.train_dataset.num_lemmas):
        if not unlabeled_dataset.has_lemma(lemma):
            continue
        try:
            tf.reset_default_graph()
            with tf.Session() as sess:
                ladder_networks = LadderNetworksExperiment(
                    labeled_train_data=data, labeled_train_target=target,
                    labeled_test_data=labeled_datasets.test_dataset.data(lemma),
                    labeled_test_target=labeled_datasets.test_dataset.target(lemma),
                    unlabeled_data=unlabeled_dataset.data(lemma, limit=args.unlabeled_data_limit),
                    layers=args.layers, denoising_cost=args.denoising_cost, epochs=args.epochs,
                    min_count=args.min_count, validation_ratio=args.validation_ratio, noise_std=args.noise_std,
                    learning_rate=0.01, evaluation_amount=1000, random_seed=args.random_seed)

                ladder_networks.run()

                pr = ladder_networks.get_results()
                pr.insert(0, 'lemma', lemma)
                prediction_results.append(pr)
                # Save the bootstrapped data
                esi, est = ladder_networks.evaluation_sentences
                evaluation_targets.extend(est)

                ul_instances = unlabeled_dataset.instances_id(lemma, limit=args.unlabeled_data_limit)
                evaluation_instances.extend(':'.join(ul_instances[idx]) for idx in esi)
        except NotEnoughSensesError:
            tqdm.write('The lemma %s doesn\'t have enough senses with at least %d occurrences'
                       % (lemma, args.min_count), file=sys.stderr)
            continue

    print('Saving results', file=sys.stderr)

    pd.DataFrame({'instance': evaluation_instances, 'predicted_target': evaluation_targets}) \
        .to_csv('%s_unlabeled_dataset_predictions.csv' % args.base_results_path, index=False)
    pd.concat(prediction_results, ignore_index=True) \
        .to_csv('%s_prediction_results.csv' % args.base_results_path, index=False, float_format='%.2e')
