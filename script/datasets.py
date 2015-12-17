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
        # Merge log features
        merged = self.df.merge(self._log_feature(), on='id')
        columns = ['feature %d' % i for i in range(1, 387)]
        merged = merged.merge(self._event_type(), on='id')
        columns.extend(['event_type_event_type %d' % i for i in range(1, 54)])
        return merged[columns]

    def _log_feature(self):
        """ Prepare log features """
        log_feature = pd.read_csv('../data/log_feature.csv')
        log_feature['volume'] = log_feature['volume'] / log_feature['volume'].max()
        lf = log_feature.groupby(['id', 'log_feature']).max()
        lf = lf.unstack(1).fillna(0)
        return lf['volume'].reset_index()

    def _event_type(self):
        """ Prepare event types """
        event_type = pd.read_csv('../data/event_type.csv')
        data = pd.get_dummies(event_type).groupby('id').sum().reset_index()
        data['event_type_event_type 16'] = data['event_type_event_type 54']
        return data

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


