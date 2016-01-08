#
# Autoencoder sandbox
#
# Hidden units, Activation, Score, Iterations, Learning rate, Train time
# 256           relu        17.99  10^4        1e-4           135m
#

import numpy as np
import tensorflow as tf

from datasets import Dataset, FEATURES_COUNT

INPUT_UNITS = FEATURES_COUNT
HIDDEN_UNITS = 256

LEARNING_RATE = 1e-4
STEPS = 5 * (10 ** 4)


class AutoEncoder(object):

    def __init__(self):
        self.x_placeholder = tf.placeholder("float", shape=[None, INPUT_UNITS])
        self.model = self._build()
        self.session = tf.Session()

    def _build(self):
        # Hidden layer
        w2 = self._weight_variable([INPUT_UNITS, HIDDEN_UNITS])
        b2 = self._bias_variable([HIDDEN_UNITS])
        l2 = tf.nn.sigmoid(tf.matmul(self.x_placeholder, w2) + b2)
        # Output layer
        w3 = self._weight_variable([HIDDEN_UNITS, INPUT_UNITS])
        b3 = self._bias_variable([INPUT_UNITS])
        l3 = tf.nn.sigmoid(tf.matmul(l2, w3) + b3)
        return tf.nn.softmax(l3)

    def loss(self, expected, predicted):
        predicted = np.minimum(predicted, 1-10**-15)
        predicted = np.maximum(predicted, 10**-15)
        return -tf.reduce_sum(expected*tf.log(predicted))

    def fit(self, X):
        """ Train network on the given data """
        cross_entropy = self.loss(self.x_placeholder, self.model)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
        init = tf.initialize_all_variables()
        self.session.run(init)
        for i in range(STEPS):
            self.session.run(train_step, feed_dict={self.x_placeholder: X})
            if i % 100 == 0:
                train_accuracy = self.check_score(X)
                print "step %d, accuracy %g" % (i, train_accuracy)

    def predict(self, X):
        return self.session.run(self.model, feed_dict={self.x_placeholder: X})

    def check_score(self, X):
        loss = self.loss(self.x_placeholder, self.model)
        score = self.session.run(loss, feed_dict={self.x_placeholder: X})
        return score/len(X)

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def train_auto_encoder():
    print('training auto encoder')
    network = AutoEncoder()
    train = Dataset.from_train()
    X = train.get_features()
    network.fit(X)
    score = network.check_score(X)
    print("Train dataset score %f" % score)


if __name__ == "__main__":
    train_auto_encoder()
    print('Done.')
