#
# Helper functions for TensorFlow solution
#

import numpy as np
import pandas as pd
from sklearn import cross_validation
from scipy import linalg


FEATURES_COUNT = 386+53+10+5
PCA_FEATURES_COUNT = 267


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
        columns = ['feature %d' % (i+1) for i in range(386)]
        merged = merged.merge(self._event_type(), on='id')
        columns.extend(['event_type_event_type %d' % (i+1) for i in range(53)])
        merged = merged.merge(self._resource_type(), on='id')
        columns.extend(['resource_type_resource_type %d' % (i+1) for i in range(10)])
        merged = merged.merge(self._severity_type(), on='id')
        columns.extend(['severity_type_severity_type %d' % (i+1) for i in range(5)])
        return merged[columns]

    def get_pca_features(self, u_reduce):
        X = self.get_features().as_matrix()
        return np.matmul(X, u_reduce)

    def pca(self):
        X = self.get_features().as_matrix()
        U, _, _ = linalg.svd(np.transpose(X))
        return U[:, 0:PCA_FEATURES_COUNT]


    def _log_feature(self):
        log_feature = pd.read_csv('../data/log_feature.csv')
        # log_feature['volume'] = log_feature['volume'] / log_feature['volume'].max()
        log_feature['volume'] = np.minimum(log_feature['volume'], 1)
        lf = log_feature.groupby(['id', 'log_feature']).max()
        lf = lf.unstack(1).fillna(0)
        return lf['volume'].reset_index()

    def _event_type(self):
        event_type = pd.read_csv('../data/event_type.csv')
        data = pd.get_dummies(event_type).groupby('id').sum().reset_index()
        data['event_type_event_type 16'] = data['event_type_event_type 54']
        return data

    def _resource_type(self):
        resource_type = pd.read_csv('../data/resource_type.csv')
        data = pd.get_dummies(resource_type).groupby('id').sum().reset_index()
        return data

    def _severity_type(self):
        severity_type = pd.read_csv('../data/severity_type.csv')
        data = pd.get_dummies(severity_type).groupby('id').sum().reset_index()
        return data

    def get_labels(self):
        self.df['severity_0'] = self.df.fault_severity.apply(lambda x: 1 if x == 0 else 0)
        self.df['severity_1'] = self.df.fault_severity.apply(lambda x: 1 if x == 1 else 0)
        self.df['severity_2'] = self.df.fault_severity.apply(lambda x: 1 if x == 2 else 0)
        return self.df[['severity_0', 'severity_1', 'severity_2']]

    def get_multi_labels(self):
        return self.df.fault_severity


def load_cross_validation(train_size=0.75):
    (train, test) = cross_validation.train_test_split(pd.read_csv('../data/train.csv'), train_size=train_size)
    return Dataset(train), Dataset(test)


def save_predictions(y, test):
    predictions = pd.DataFrame(y, columns=['predict_0', 'predict_1', 'predict_2'])
    predictions['id'] = test['id']
    predictions[['id', 'predict_0', 'predict_1', 'predict_2']].to_csv('../data/solution.csv', index=False)


def test():
    train = Dataset.from_train()
    logs = train._log_feature()
    # Check if log features are normalized
    print(logs.describe())
    print('Done.')

if __name__ == "__main__":
    test()
