#
# Deep Belief Network
#
# Some insights:
#  - relu often have problem with exploding gradient
#  - relu6 works ok. Worth trying long learning with: learning_rate=1e-4, keep_prob=0.2
#  - elu
#

from collections import namedtuple

import numpy as np
import tensorflow as tf

from datasets import Dataset, save_predictions, load_cross_validation, FEATURES_COUNT

Params = namedtuple('Params', [
    'learning_rate',
    'steps',
    'stop_on_test_score',
    'keep_prob'                    # Dropout probability
])

INPUT_UNITS = FEATURES_COUNT
HIDDEN_UNITS_1 = 64
HIDDEN_UNITS_2 = 32
OUTPUT_CLASSES = 3


class DeepBelieflNetwork(object):
    """
    Deep Belief Network with the following architecture:
        * Input layer with INPUT_UNITS units
        * Dense layer with 64 units
        * Dropout layer with 64 units
        * Dense layer with 32 units
        * Dropout layer with 32 units
        * Output layer: Softmax with 3 units
    """

    def __init__(self, params):
        self.params = params
        self.x_placeholder = tf.placeholder("float", shape=[None, INPUT_UNITS])
        self.y_placeholder = tf.placeholder("float", shape=[None, OUTPUT_CLASSES])
        self.keep_prob = tf.placeholder("float")
        self.model = self._build()
        self.session = tf.Session()

    def _build(self):
        # First hidden layer
        w2 = self._weight_variable([INPUT_UNITS, HIDDEN_UNITS_1])
        b2 = self._bias_variable([HIDDEN_UNITS_1])
        l2 = tf.nn.relu6(tf.matmul(self.x_placeholder, w2) + b2)
        l2_drop = tf.nn.dropout(l2, self.keep_prob)
        # Second hidden layer
        w3 = self._weight_variable([HIDDEN_UNITS_1, HIDDEN_UNITS_2])
        b3 = self._bias_variable([HIDDEN_UNITS_2])
        l3 = tf.nn.relu6(tf.matmul(l2_drop, w3) + b3)
        l3_drop = tf.nn.dropout(l3, self.keep_prob)
        # Output layer
        w5 = self._weight_variable([HIDDEN_UNITS_2, OUTPUT_CLASSES])
        b5 = self._bias_variable([OUTPUT_CLASSES])
        l5 = tf.nn.relu6(tf.matmul(l3_drop, w5) + b5)
        return tf.nn.softmax(l5)

    def loss(self, expected, predicted):
        predicted = np.minimum(predicted, 1-10**-15)
        predicted = np.maximum(predicted, 10**-15)
        return -tf.reduce_sum(expected*tf.log(predicted))

    def fit(self, X, y, X2=None, y2=None):
        """ Train network on given data """
        cross_entropy = self.loss(self.y_placeholder, self.model)
        train_step = tf.train.AdamOptimizer(self.params.learning_rate).minimize(cross_entropy)
        init = tf.initialize_all_variables()
        self.session.run(init)
        last_score = 1
        test_accuracy = 0
        for i in range(self.params.steps):
            self.session.run(train_step, feed_dict={self.x_placeholder: X, self.y_placeholder: y,
                                                    self.keep_prob: self.params.keep_prob})
            if i % 100 == 0:
                train_accuracy = self.check_score(X, y)
                if X2 is not None:
                    test_accuracy = self.check_score(X2, y2)
                print "step %d, training accuracy %g (test: %g)" % (i, train_accuracy, test_accuracy)
                if test_accuracy < 0.6 and test_accuracy < last_score:
                    make_submission(self.model)
                    last_score = test_accuracy
                # if self.params.stop_on_test_score and test_accuracy > last_score:
                #     print('Accuracy is getting worse. Stop learning')
                #     break
                # else:
                #     last_score = test_accuracy

    def predict(self, X):
        Y = self.session.run(self.model, feed_dict={self.x_placeholder: X, self.keep_prob: 1})
        return Y

    def check_score(self, X, Y):
        loss = self.loss(self.y_placeholder, self.model)
        score = self.session.run(loss, feed_dict={self.x_placeholder: X, self.y_placeholder: Y, self.keep_prob: 1})
        return score/len(X)

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def make_submission(network):
    print('Prepare submission')
    test = Dataset.from_test()
    X2 = test.get_features()
    predictions = network.predict(X2)
    save_predictions(predictions, test.df)


def cross_validate(params):
    print('Cross validate with params')
    print(params)
    network = DeepBelieflNetwork(params)
    train, test = load_cross_validation(0.8)
    X = train.get_features()
    Y = train.get_labels()
    X2 = test.get_features()
    Y2 = test.get_labels()
    network.fit(X, Y, X2, Y2)
    score = network.check_score(X, Y)
    print("Train dataset score %f" % score)
    score = network.check_score(X2, Y2)
    print("test dataset score %f" % score)
    return network


# (0.559475, 0.585941) Kaggle: 0.63373 (Best so far)
PARAMS1 = Params(learning_rate=1e-4, steps=10**3, stop_on_test_score=False, keep_prob=0.4)
# (0.638753, 0.648865) 20 000 steps, alpha=1e-4, keep_prob=0.2
PARAMS = Params(learning_rate=1e-5, steps=2*(10**5), stop_on_test_score=False, keep_prob=0.3)

if __name__ == "__main__":
    model = cross_validate(PARAMS)
    # make_submission(model)
    print('Done.')
