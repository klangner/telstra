#
# Neural network solution
#

import sys
from collections import namedtuple

import numpy as np
import tensorflow as tf
from sklearn import cross_validation

from datasets import Dataset, save_predictions, load_cross_validation, PCA_FEATURES_COUNT, FEATURES_COUNT

Params = namedtuple('Params', [
    'learning_rate',
    'activation_function',          # 'sigmoid or 'relu'
    'steps',                        # Log10 scale
    'stop_on_test_score',           # Stop training if score on test set is getting bigger
    'pca',                          # Use PCA on feature vestors if True
    'keep_prob',                    # Dropout probability
    'hidden_units',                 # Number of hidden units
    'normalization'                 # Normalization coefficient
])

OUTPUT_CLASSES = 3


class NeuralNetwork(object):

    def __init__(self, params):
        self.params = params
        self.feature_count = PCA_FEATURES_COUNT if params.pca else FEATURES_COUNT
        self.x_placeholder = tf.placeholder("float", shape=[None, self.feature_count])
        self.y_placeholder = tf.placeholder("float", shape=[None, OUTPUT_CLASSES])
        self.model = self._build()
        self.session = tf.Session()
        init = tf.initialize_all_variables()
        self.session.run(init)

    def _build(self):
        """ NN with single hidden layer """
        self.w2 = self._weight_variable([self.feature_count, self.params.hidden_units])
        b2 = self._bias_variable([self.params.hidden_units])
        if self.params.activation_function == 'sigmoid':
            l2 = tf.nn.sigmoid(tf.matmul(self.x_placeholder, self.w2) + b2)
        else:
            l2 = tf.nn.relu(tf.matmul(self.x_placeholder, self.w2) + b2)
        self.w3 = self._weight_variable([self.params.hidden_units, OUTPUT_CLASSES])
        b3 = self._bias_variable([OUTPUT_CLASSES])
        if self.params.activation_function == 'sigmoid':
            l3 = tf.nn.sigmoid(tf.matmul(l2, self.w3) + b3)
        else:
            l3 = tf.nn.relu(tf.matmul(l2, self.w3) + b3)
        self.keep_prob = tf.placeholder("float")
        l3_drop = tf.nn.dropout(l3, self.keep_prob)
        return tf.nn.softmax(l3_drop)

    def loss(self, expected, predicted):
        predicted = np.minimum(predicted, 1-10**-15)
        predicted = np.maximum(predicted, 10**-15)
        return -tf.reduce_sum(expected*tf.log(predicted))

    def normalized_loss(self, expected, predicted):
        predicted = np.minimum(predicted, 1-10**-15)
        predicted = np.maximum(predicted, 10**-15)
        w2 = tf.reduce_sum(tf.pow(self.w2, 2))
        w3 = tf.reduce_sum(tf.pow(self.w3, 2))
        l2 = self.params.normalization*(w2*w3)/self.params.hidden_units
        return -tf.reduce_sum(expected*tf.log(predicted)) + l2

    def fit(self, X, y, X2=None, y2=None):
        """ Train network on given data """
        normalized_cross_entropy = self.normalized_loss(self.y_placeholder, self.model)
        cross_entropy = self.loss(self.y_placeholder, self.model)
        train_step = tf.train.GradientDescentOptimizer(self.params.learning_rate).minimize(normalized_cross_entropy)
        last_score = 10 ** 6
        test_accuracy = 0
        for i in range(10 ** self.params.steps):
            (A, _, label_a, _) = cross_validation.train_test_split(X, y, train_size=100)
            self.session.run(train_step, feed_dict={self.x_placeholder: A, self.y_placeholder: label_a, self.keep_prob: self.params.keep_prob})
            if i % 1000 == 0:
                train_accuracy = self.session.run(cross_entropy,
                                                  feed_dict={self.x_placeholder: X, self.y_placeholder: y, self.keep_prob: 1})/len(X)
                if X2 is not None:
                    test_accuracy = self.session.run(cross_entropy,
                                                     feed_dict={self.x_placeholder: X2, self.y_placeholder: y2, self.keep_prob: 1})/len(X2)
                print "step %d, training accuracy %g (test: %g)" % (i, train_accuracy, test_accuracy)
                if self.params.stop_on_test_score and test_accuracy > last_score:
                    print('Accuracy is getting worse. Stop learning')
                    break
                else:
                    last_score = test_accuracy

    def predict(self, X):
        Y = self.session.run(self.model, feed_dict={self.x_placeholder: X, self.keep_prob: 1})
        return Y

    def check_score(self, X, Y):
        loss = self.loss(self.y_placeholder, self.model)
        return self.session.run(loss, feed_dict={self.x_placeholder: X, self.y_placeholder: Y, self.keep_prob: 1})

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def make_submission(network, params, u):
    print('Prepare submission')
    test = Dataset.from_test()
    if params.pca:
        X2 = test.get_pca_features(u)
    else:
        X2 = test.get_features()
    predictions = network.predict(X2)
    save_predictions(predictions, test.df)


def cross_validate(params):
    print('Cross validate with params')
    print(params)
    network = NeuralNetwork(params)
    train, test = load_cross_validation(0.8)
    u = train.pca()
    if params.pca:
        X = train.get_pca_features(u)
    else:
        X = train.get_features()
    Y = train.get_labels()
    if params.pca:
        X2 = test.get_pca_features(u)
    else:
        X2 = test.get_features()
    Y2 = test.get_labels()
    network.fit(X, Y, X2, Y2)
    score = network.check_score(X, Y)
    print("Train dataset score %f" % (score/len(X)))
    score = network.check_score(X2, Y2)
    print("test dataset score %f" % (score/len(X2)))
    make_submission(network, params, u)


def prepare_submission(params):
    print('Prepare submission with params')
    print(params)
    network = NeuralNetwork(params)
    train = Dataset.from_train()
    u = train.pca()
    if params.pca:
        X = train.get_pca_features(u)
    else:
        X = train.get_features()
    Y = train.get_labels()
    network.fit(X, Y)
    score = network.check_score(X, Y)
    print('Train dataset score %f' % (score/len(X)))
    make_submission(network, params, u)


# 10^5 steps is around 10 minutes of training
# Score is given as (train_score, test_score)

# (0.819590, 0.825475) Fast with sigmoid functions
FAST_SIGMOID_1 = Params(learning_rate=0.0003, steps=5, activation_function='sigmoid', stop_on_test_score=False,
                        pca=True, keep_prob=1, hidden_units=10, normalization=0)

# (0.475820, 0.669871) Fast with relu functions
FAST_RELU_1 = Params(learning_rate=0.0003, steps=5, activation_function='relu', stop_on_test_score=False,
                     pca=True, keep_prob=1, hidden_units=10, normalization=0)

# (0.592587, 0.606354) Fast with relu functions. Stop when over fits
FAST_RELU_2 = Params(learning_rate=0.0003, steps=5, activation_function='relu', stop_on_test_score=True,
                     pca=True, keep_prob=1, hidden_units=10, normalization=0)

FAST_RELU_10 = Params(learning_rate=0.0003, steps=5, activation_function='relu', stop_on_test_score=False,
                      pca=True, keep_prob=1, hidden_units=10, normalization=0.6)

# 0.600571, 0.609436 After submission: 0.64360
FAST_RELU_30 = Params(learning_rate=0.0003, steps=5, activation_function='relu', stop_on_test_score=False,
                      pca=True, keep_prob=1, hidden_units=30, normalization=0.9)

#
FAST_RELU_50 = Params(learning_rate=0.0003, steps=5, activation_function='relu', stop_on_test_score=False,
                      pca=True, keep_prob=1, hidden_units=50, normalization=0.6)


# (0.641099, 0.66908) Submission
SUBMISSION_PARAM_10 = Params(learning_rate=0.0003, steps=5, activation_function='relu', stop_on_test_score=False,
                           pca=True, keep_prob=1, hidden_units=30, normalization=0)

if __name__ == "__main__":
    if 'submit' in sys.argv:
        prepare_submission(FAST_RELU_30)
    else:
        cross_validate(FAST_RELU_30)
    print('Done.')
