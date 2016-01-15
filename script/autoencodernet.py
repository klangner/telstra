#
# Autoencoder sandbox
#
# Hidden units, Activation, Score, Iterations, Learning rate, Train time
# 256           relu        17.99  10^4        1e-4           135m
#
# Best score after several hours of training: 17.533761

import numpy as np
import sys
import tensorflow as tf
from sklearn import cross_validation

import helpers
from datasets import Dataset, FEATURES_COUNT

INPUT_UNITS = FEATURES_COUNT
ENCODED_FEATURES = 256
HIDDEN_UNITS_2 = 10
OUTPUT_CLASSES = 3

# LEARNING_RATE = 1e-4
# STEPS = 10 ** 3
# KEEP_PROB = 0.5

LEARNING_RATE = 1e-4
STEPS = 10 ** 5
KEEP_PROB = 0.2

SESSION_FILE = '../data/autoencoder.session'


class AutoEncoder(object):

    def __init__(self):
        self.x_placeholder = tf.placeholder("float", shape=[None, INPUT_UNITS])
        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder(self._encoder)
        self.session = tf.Session()
        self._saver = tf.train.Saver()

    def _build_encoder(self):
        weights = helpers.weight_variable([INPUT_UNITS, ENCODED_FEATURES])
        biases = helpers.bias_variable([ENCODED_FEATURES])
        return tf.nn.relu(tf.matmul(self.x_placeholder, weights) + biases)

    def _build_decoder(self, encoder):
        weights = helpers.weight_variable([ENCODED_FEATURES, INPUT_UNITS])
        biases = helpers.bias_variable([INPUT_UNITS])
        l3 = tf.nn.relu(tf.matmul(encoder, weights) + biases)
        return tf.nn.softmax(l3)

    def loss(self, expected, predicted):
        predicted = np.minimum(predicted, 1-10**-15)
        predicted = np.maximum(predicted, 10**-15)
        return -tf.reduce_sum(expected*tf.log(predicted))

    def fit_encoder(self, train_data, test_data, restore=False):
        """ Train network on the given data """
        cross_entropy = self.loss(self.x_placeholder, self._decoder)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
        if restore:
            self.restore_session()
        else:
            init = tf.initialize_all_variables()
            self.session.run(init)
        X = train_data.get_features()
        for i in range(STEPS):
            # X, _ = train_data.next_batch()
            self.session.run(train_step, feed_dict={self.x_placeholder: X})
            if i % 100 == 0:
                train_accuracy = self.check_encoder_score(train_data)
                test_accuracy = self.check_encoder_score(test_data)
                print "step %d, train accuracy %g (test: %g)" % (i, train_accuracy, test_accuracy)
                self.save_session()
        self.save_session()

    def check_encoder_score(self, data):
        loss = self.loss(self.x_placeholder, self._decoder)
        score = self.session.run(loss, feed_dict={self.x_placeholder: data.get_features()})
        return score/data.size()

    def encode(self, X):
        return self.session.run(self._encoder, feed_dict={self.x_placeholder: X})

    def save_session(self):
        self._saver.save(self.session, SESSION_FILE)

    def restore_session(self):
        init = tf.initialize_all_variables()
        self.session.run(init)
        self._saver.restore(self.session, SESSION_FILE)


class NeuralNet(object):

    def __init__(self):
        self.x_placeholder = tf.placeholder("float", shape=[None, ENCODED_FEATURES])
        self.y_placeholder = tf.placeholder("float", shape=[None, OUTPUT_CLASSES])
        self.keep_prob = tf.placeholder("float")
        self._model = self._build_net()
        self.session = tf.Session()

    def _build_net(self):
        w2 = helpers.weight_variable([ENCODED_FEATURES, HIDDEN_UNITS_2])
        b2 = helpers.bias_variable([HIDDEN_UNITS_2])
        l2 = tf.nn.relu6(tf.matmul(self.x_placeholder, w2) + b2)
        l2_drop = tf.nn.dropout(l2, self.keep_prob)
        # Output layer
        w5 = helpers.weight_variable([HIDDEN_UNITS_2, OUTPUT_CLASSES])
        b5 = helpers.bias_variable([OUTPUT_CLASSES])
        l5 = tf.nn.relu6(tf.matmul(l2_drop, w5) + b5)
        return tf.nn.softmax(l5)

    @staticmethod
    def loss(expected, predicted):
        predicted = np.minimum(predicted, 1-10**-15)
        predicted = np.maximum(predicted, 10**-15)
        return -tf.reduce_sum(expected*tf.log(predicted))

    def fit(self, features, labels):
        """ Train network on the given data """
        cross_entropy = self.loss(self.y_placeholder, self._model)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
        init = tf.initialize_all_variables()
        self.session.run(init)
        last_score = 1
        X, X2, y, y2 = cross_validation.train_test_split(features, labels, train_size=0.8)
        for i in range(STEPS):
            self.session.run(train_step, feed_dict={self.x_placeholder: X,
                                                    self.y_placeholder: y,
                                                    self.keep_prob: KEEP_PROB})
            if i % 100 == 0:
                train_accuracy = self.check_score(X, y)
                test_accuracy = self.check_score(X2, y2)
                print "step %d, train accuracy %g (test: %g)" % (i, train_accuracy, test_accuracy)
                if test_accuracy < 0.6 and test_accuracy < last_score:
                    print('make_submission(self.model)')
                    last_score = test_accuracy

    def check_score(self, X, y):
        loss = self.loss(self.y_placeholder, self._model)
        score = self.session.run(loss, feed_dict={self.x_placeholder: X,
                                                  self.y_placeholder: y,
                                                  self.keep_prob: 1})
        return score/len(X)

    def errors(self, X, labels):
        predictions = self.session.run(self._model, feed_dict={self.x_placeholder: X})
        e = np.argmax(labels, axis=1)
        p = np.argmax(predictions, axis=1)
        return (np.not_equal(p, e).sum()*100.0)/len(p)


def train_auto_encoder(restore):
    print('Training auto encoder')
    network = AutoEncoder()
    train_data = Dataset.from_train()
    test_data = Dataset.from_test()
    network.fit_encoder(train_data, test_data, restore=restore)


def train_nn(restore):
    print('Training neural net')
    encoder = AutoEncoder()
    encoder.restore_session()
    train_data = Dataset.from_train()
    X = encoder.encode(train_data.get_features())
    y = train_data.get_labels()
    nn = NeuralNet()
    nn.fit(X, y)


if __name__ == "__main__":
    if 'nn' in sys.argv:
        train_nn(3)
    elif 'restore' in sys.argv:
        train_auto_encoder(True)
    else:
        train_auto_encoder(False)
    print('Done.')
