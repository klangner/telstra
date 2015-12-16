#
# Main function which calculates score on cross_validation and prepares solution.
#
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation


def load_train():
    return pd.read_csv('../data/train.csv')


def load_test():
    return pd.read_csv('../data/test.csv')


def build_features(df):
    log_feature = pd.read_csv('../data/log_feature.csv')
    lf = log_feature.groupby(['id', 'log_feature']).max()
    lf = lf.unstack(1).fillna(0)
    lf = lf['volume'].reset_index()
    df = df.merge(lf, on='id')
    columns = ['feature %d' % i for i in range(1, 387)]
    return df[columns]


def extract_labels(df):
    df['severity_0'] = df.fault_severity.apply(lambda x: 1 if x == 0 else 0)
    df['severity_1'] = df.fault_severity.apply(lambda x: 1 if x == 1 else 0)
    df['severity_2'] = df.fault_severity.apply(lambda x: 1 if x == 2 else 0)
    return df[['severity_0', 'severity_1', 'severity_2']]


def prepare_solution():
    train = load_train()
    X = build_features(train)
    Y = extract_labels(train)
    rf = RandomForestRegressor(n_jobs=-1)
    model = rf.fit(X, Y)
    print('Train score: %f' % loss(Y, model.predict(X)))
    test = load_test()
    X2 = build_features(test)
    Y2 = model.predict(X2)
    save_predictions(Y2, test)


def save_predictions(y, test):
    predictions = pd.DataFrame(y, columns=['predict_0', 'predict_1', 'predict_2'])
    predictions['id'] = test['id']
    predictions[['id', 'predict_0', 'predict_1', 'predict_2']].to_csv('../data/solution.csv', index=False)


def cross_validate():
    (train, test) = cross_validation.train_test_split(load_train())
    X = build_features(train)
    Y = extract_labels(train)
    X2 = build_features(test)
    Y2 = extract_labels(test)
    rf = RandomForestRegressor(n_jobs=-1)
    model = rf.fit(X, Y)
    print('Cross validation score: %f' % loss(Y2, model.predict(X2)))


def loss(expected, predicted):
    predicted = np.minimum(predicted, 1-10**-15)
    predicted = np.maximum(predicted, 10**-15)
    ps = np.multiply(expected, np.log(predicted))
    return -ps.sum().sum()/len(expected)


prepare_solution()
cross_validate()
