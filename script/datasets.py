#
# Helper functions for TensorFlow solution
#

import pandas as pd
from sklearn import cross_validation


class Dataset(object):

    def __init__(self, data):
        self.df = data

    @staticmethod
    def from_train():
        return Dataset(pd.read_csv('../data/train.csv'))

    @staticmethod
    def from_test():
        return Dataset(pd.read_csv('../data/test.csv'))

    def get_features(self):
        # Prepare log features
        log_feature = pd.read_csv('../data/log_feature.csv')
        log_feature['volume'] = log_feature['volume'] / log_feature['volume'].max()
        lf = log_feature.groupby(['id', 'log_feature']).max()
        lf = lf.unstack(1).fillna(0)
        lf = lf['volume'].reset_index()
        # Merge log features
        merged = self.df.merge(lf, on='id')
        # Select DataFrame columns
        columns = ['feature %d' % i for i in range(1, 387)]
        return merged[columns]

    def get_labels(self):
        self.df['severity_0'] = self.df.fault_severity.apply(lambda x: 1 if x == 0 else 0)
        self.df['severity_1'] = self.df.fault_severity.apply(lambda x: 1 if x == 1 else 0)
        self.df['severity_2'] = self.df.fault_severity.apply(lambda x: 1 if x == 2 else 0)
        return self.df[['severity_0', 'severity_1', 'severity_2']]


def load_cross_validation():
    (train, test) = cross_validation.train_test_split(pd.read_csv('../data/train.csv'), train_size=0.75)
    return Dataset(train), Dataset(test)


def save_predictions(y, test):
    predictions = pd.DataFrame(y, columns=['predict_0', 'predict_1', 'predict_2'])
    predictions['id'] = test['id']
    predictions[['id', 'predict_0', 'predict_1', 'predict_2']].to_csv('../data/solution.csv', index=False)


