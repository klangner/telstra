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

import helpers
from datasets import Dataset, FEATURES_COUNT

INPUT_UNITS = FEATURES_COUNT
HIDDEN_UNITS_1 = 256
HIDDEN_UNITS_2 = 30
OUTPUT_CLASSES = 3

LEARNING_RATE = 1e-4
STEPS = 10 ** 3
SESSION_FILE = '../data/autoencoder.session'


class AutoEncoderNet(object):

    def __init__(self):
        self.x_placeholder = tf.placeholder("float", shape=[None, INPUT_UNITS])
        self.keep_prob = tf.placeholder("float")
        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder(self._encoder)
        self.session = tf.Session()
        self._saver = tf.train.Saver()

    def _build_encoder(self):
        weights = helpers.weight_variable([INPUT_UNITS, HIDDEN_UNITS_1])
        biases = helpers.bias_variable([HIDDEN_UNITS_1])
        return tf.nn.relu(tf.matmul(self.x_placeholder, weights) + biases)

    def _build_decoder(self, encoder):
        weights = helpers.weight_variable([HIDDEN_UNITS_1, INPUT_UNITS])
        biases = helpers.bias_variable([INPUT_UNITS])
        l3 = tf.nn.relu(tf.matmul(encoder, weights) + biases)
        return tf.nn.softmax(l3)

    def _build_net(self, encoder):
        # Second hidden layer
        w3 = helpers.weight_variable([HIDDEN_UNITS_1, HIDDEN_UNITS_2])
        b3 = helpers.bias_variable([HIDDEN_UNITS_2])
        l3 = tf.nn.relu6(tf.matmul(encoder, w3) + b3)
        l3_drop = tf.nn.dropout(l3, self.keep_prob)
        # Output layer
        w5 = helpers.weight_variable([HIDDEN_UNITS_2, OUTPUT_CLASSES])
        b5 = helpers.bias_variable([OUTPUT_CLASSES])
        l5 = tf.nn.relu6(tf.matmul(l3_drop, w5) + b5)
        return tf.nn.softmax(l5)


    def loss(self, expected, predicted):
        predicted = np.minimum(predicted, 1-10**-15)
        predicted = np.maximum(predicted, 10**-15)
        return -tf.reduce_sum(expected*tf.log(predicted))

    def fit_encoder(self, train_data, test_data, restore=False):
        """ Train network on the given data """
        cross_entropy = self.loss(self.x_placeholder, self._decoder)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
        init = tf.initialize_all_variables()
        self.session.run(init)
        if restore:
            self.restore_session()
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
        self._saver.restore(self.session, SESSION_FILE)


def train_auto_encoder(restore):
    print('training auto encoder')
    network = AutoEncoderNet()
    train_data = Dataset.from_train()
    test_data = Dataset.from_test()
    network.fit_encoder(train_data, test_data, restore=restore)
    # score = network.check_score(train_data)
    # print("Train dataset score %f" % score)


if __name__ == "__main__":
    if 'restore' in sys.argv:
        train_auto_encoder(True)
    else:
        train_auto_encoder(False)
    print('Done.')
