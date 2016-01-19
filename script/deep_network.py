#
# Deep Belief Network
#
# Some insights:
#  - relu often have problem with exploding gradient
#  - relu6 works ok. Worth trying long learning with: learning_rate=1e-4, keep_prob=0.2
#  - elu
#

from collections import namedtuple

import tensorflow as tf

from datasets import *
import helpers


SESSION_FILE = '../data/deep_network.session'

Params = namedtuple('Params', [
    'learning_rate',
    'steps',
    'keep_prob'                    # Dropout probability
])

INPUT_UNITS = FEATURES_COUNT
HIDDEN_UNITS_1 = 64
HIDDEN_UNITS_2 = 32
OUTPUT_CLASSES = 3


class DeepNetwork(object):
    """
    Deep Network with the following architecture:
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
        self._saver = tf.train.Saver()

    def _build(self):
        # First hidden layer
        w2 = helpers.weight_variable([INPUT_UNITS, HIDDEN_UNITS_1])
        b2 = helpers.bias_variable([HIDDEN_UNITS_1])
        l2 = tf.nn.relu6(tf.matmul(self.x_placeholder, w2) + b2)
        l2_drop = tf.nn.dropout(l2, self.keep_prob)
        # Second hidden layer
        w3 = helpers.weight_variable([HIDDEN_UNITS_1, HIDDEN_UNITS_2])
        b3 = helpers.bias_variable([HIDDEN_UNITS_2])
        l3 = tf.nn.relu6(tf.matmul(l2_drop, w3) + b3)
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

    def fit(self, train_data, cv_data = None):
        """ Train network on the given data """
        cross_entropy = self.loss(self.y_placeholder, self.model)
        train_step = tf.train.AdamOptimizer(self.params.learning_rate).minimize(cross_entropy)
        init = tf.initialize_all_variables()
        self.session.run(init)
        best_score = 1
        not_improvement_count = 0
        for i in range(self.params.steps):
            X, y = train_data.next_batch()
            self.session.run(train_step, feed_dict={self.x_placeholder: X, self.y_placeholder: y,
                                                    self.keep_prob: self.params.keep_prob})
            if i % 1000 == 0:
                train_accuracy = self.check_score(train_data)
                if cv_data is not None:
                    test_accuracy = self.check_score(cv_data)
                print "step %d, training accuracy %g (test: %g)" % (i, train_accuracy, test_accuracy)
                if test_accuracy < 0.63:
                    if test_accuracy < best_score:
                        self.save_session()
                        best_score = test_accuracy
                        not_improvement_count = 0
                    elif not_improvement_count < 3:
                        not_improvement_count += 1
                    else:
                        break

    def predict(self, X):
        Y = self.session.run(self.model, feed_dict={self.x_placeholder: X, self.keep_prob: 1})
        return Y

    def check_score(self, data):
        loss = self.loss(self.y_placeholder, self.model)
        score = self.session.run(loss, feed_dict={self.x_placeholder: data.get_features(),
                                                  self.y_placeholder: data.get_labels(),
                                                  self.keep_prob: 1})
        return score/data.size()

    def save_session(self):
        self._saver.save(self.session, SESSION_FILE)

    def restore_session(self):
        self._saver.restore(self.session, SESSION_FILE)


def make_submission(network):
    print('Prepare submission')
    data = Dataset.from_test()
    predictions = network.predict(data.get_features())
    save_predictions(predictions, data.df)


def cross_validate(params):
    print('Cross validate with params')
    print(params)
    network = DeepNetwork(params)
    train, test = load_cross_validation(0.8)
    network.fit(train, test)
    network.restore_session()
    score = network.check_score(train)
    print("Train dataset score %f" % score)
    score = network.check_score(test)
    print("test dataset score %f" % score)
    return network


# (0.559475, 0.585941) Kaggle: 0.63373 (Best so far)
PARAMS1 = Params(learning_rate=1e-4, steps=10**5, keep_prob=1)

# (0.638753, 0.648865) 20 000 steps, alpha=1e-4, keep_prob=0.2
PARAMS = Params(learning_rate=1e-5, steps=2*(10**5), keep_prob=0.4)

if __name__ == "__main__":
    model = cross_validate(PARAMS1)
    make_submission(model)
    print('Done.')
