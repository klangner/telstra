#
# Neural network solution
#

import tensorflow as tf
import numpy as np
from sklearn import cross_validation
from datasets import Dataset, save_predictions, load_cross_validation


FEATURES_COUNT = 386
HIDDEN_NEURON_COUNT = 15
OUTPUT_CLASSES = 3


class ReluNetwork(object):

    def __init__(self, train_steps=10**3):
        self.steps = train_steps
        self.x_placeholder = tf.placeholder("float", shape=[None, FEATURES_COUNT])
        self.y_placeholder = tf.placeholder("float", shape=[None, OUTPUT_CLASSES])
        self.model = self._build()
        self.session = tf.Session()
        init = tf.initialize_all_variables()
        self.session.run(init)

    def _build(self):
        w2 = self._weight_variable([FEATURES_COUNT, HIDDEN_NEURON_COUNT])
        b2 = self._bias_variable([HIDDEN_NEURON_COUNT])
        l2 = tf.nn.relu(tf.matmul(self.x_placeholder, w2) + b2)
        w3 = self._weight_variable([HIDDEN_NEURON_COUNT, OUTPUT_CLASSES])
        b3 = self._bias_variable([OUTPUT_CLASSES])
        l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
        return tf.nn.softmax(l3)

    def loss(self, expected, predicted):
        predicted = np.minimum(predicted, 1-10**-15)
        predicted = np.maximum(predicted, 10**-15)
        return -tf.reduce_sum(expected*tf.log(predicted))

    def fit(self, X, y):
        """ Train network on given data """
        cross_entropy = self.loss(self.y_placeholder, self.model)
        train_step = tf.train.GradientDescentOptimizer(0.0003).minimize(cross_entropy)
        for i in range(self.steps):
            (A, _, label_a, _) = cross_validation.train_test_split(X, y, train_size=100)
            self.session.run(train_step, feed_dict={self.x_placeholder: A, self.y_placeholder: label_a})
            if i % (self.steps/10) == 0:
                train_accuracy = self.session.run(cross_entropy, feed_dict={self.x_placeholder: X, self.y_placeholder: y})
                print "step %d, training accuracy %g" % (i, train_accuracy)

    def predict(self, X):
        Y = self.session.run(self.model, feed_dict={self.x_placeholder: X})
        return Y

    def check_score(self, X, Y):
        loss = self.loss(self.y_placeholder, self.model)
        return self.session.run(loss, feed_dict={self.x_placeholder: X, self.y_placeholder: Y})

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def cross_validate():
    print('Cross validate neural network with %d hidden units' % HIDDEN_NEURON_COUNT)
    network = ReluNetwork()
    train, test = load_cross_validation()
    X = train.get_features()
    Y = train.get_labels()
    X2 = test.get_features()
    Y2 = test.get_labels()
    network.fit(X, Y)
    score = network.check_score(X, Y)
    print("Train dataset score %f" % (score/len(X)))
    score = network.check_score(X2, Y2)
    print("test dataset score %f" % (score/len(X2)))


def prepare_submission():
    print('Solution: Neural Network with %d hidden units' % HIDDEN_NEURON_COUNT)
    network = ReluNetwork(train_steps=10**5)
    train = Dataset.from_train()
    test = Dataset.from_test()
    X = train.get_features()
    Y = train.get_labels()
    X2 = test.get_features()
    network.fit(X, Y)
    score = network.check_score(X, Y)
    print('Train dataset score %f' % (score/len(X)))
    predictions = network.predict(X2)
    save_predictions(predictions, test.df)


cross_validate()
# prepare_submission()
