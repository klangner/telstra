#
# Neural network solution
#

import sys
import tensorflow as tf
import numpy as np
from sklearn import cross_validation
from datasets import Dataset, save_predictions, load_cross_validation


FEATURES_COUNT = 386+53+10+5
HIDDEN_NEURON_COUNT = 10
OUTPUT_CLASSES = 3
LEARNING_RATE = 0.0003


class NeuralNetwork(object):

    def __init__(self, train_steps=10**3):
        self.steps = train_steps
        self.x_placeholder = tf.placeholder("float", shape=[None, FEATURES_COUNT])
        self.y_placeholder = tf.placeholder("float", shape=[None, OUTPUT_CLASSES])
        self.model = self._build()
        self.session = tf.Session()
        init = tf.initialize_all_variables()
        self.session.run(init)

    def _build(self):
        """ NN with single hidden layer """
        w2 = self._weight_variable([FEATURES_COUNT, HIDDEN_NEURON_COUNT])
        b2 = self._bias_variable([HIDDEN_NEURON_COUNT])
        l2 = tf.nn.relu(tf.matmul(self.x_placeholder, w2) + b2)
        w3 = self._weight_variable([HIDDEN_NEURON_COUNT, OUTPUT_CLASSES])
        b3 = self._bias_variable([OUTPUT_CLASSES])
        l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
        return tf.nn.softmax(l3)

    def _build2(self):
        """ NN with 2 hidden layers. (Doesn't work any better) """
        w2 = self._weight_variable([FEATURES_COUNT, HIDDEN_NEURON_COUNT])
        b2 = self._bias_variable([HIDDEN_NEURON_COUNT])
        l2 = tf.nn.relu(tf.matmul(self.x_placeholder, w2) + b2)
        w3 = self._weight_variable([HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT])
        b3 = self._bias_variable([HIDDEN_NEURON_COUNT])
        l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
        w4 = self._weight_variable([HIDDEN_NEURON_COUNT, OUTPUT_CLASSES])
        b4 = self._bias_variable([OUTPUT_CLASSES])
        l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
        return tf.nn.softmax(l4)

    def loss(self, expected, predicted):
        predicted = np.minimum(predicted, 1-10**-15)
        predicted = np.maximum(predicted, 10**-15)
        return -tf.reduce_sum(expected*tf.log(predicted))

    def fit(self, X, y):
        """ Train network on given data """
        cross_entropy = self.loss(self.y_placeholder, self.model)
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
        last_score = 10 ** 6
        for i in range(self.steps):
            (A, _, label_a, _) = cross_validation.train_test_split(X, y, train_size=100)
            self.session.run(train_step, feed_dict={self.x_placeholder: A, self.y_placeholder: label_a})
            if i % 1000 == 0:
                train_accuracy = self.session.run(cross_entropy, feed_dict={self.x_placeholder: X, self.y_placeholder: y})
                print "step %d, training accuracy %g" % (i, train_accuracy)
                if train_accuracy > last_score:
                    print('Accuracy is getting worse. Stop learning')
                    break
                else:
                    last_score = train_accuracy

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
    print('Cross validate with %d hidden units, %s features and learning rate=%f' %
          (HIDDEN_NEURON_COUNT, FEATURES_COUNT, LEARNING_RATE))
    network = NeuralNetwork(train_steps=10 ** 5)
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
    print('Solution with %d hidden units, %s features and learning rate=%f' %
          (HIDDEN_NEURON_COUNT, FEATURES_COUNT, LEARNING_RATE))
    network = NeuralNetwork(train_steps=10 ** 7)
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


if __name__ == "__main__":
    if 'submit' in sys.argv:
        print('Prepare submission')
        prepare_submission()
    else:
        print('Cross validate')
        cross_validate()
