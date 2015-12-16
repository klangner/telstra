#
# Main function which calculates score on cross_validation and prepares solution.
#
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


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
    test = load_test()
    X2 = build_features(test)
    Y2 = model.predict(X2)
    predictions = pd.DataFrame(Y2, columns=['predict_0', 'predict_1', 'predict_2'])
    predictions['id'] = test['id']
    predictions[['id', 'predict_0', 'predict_1', 'predict_2']].to_csv('../data/solution.csv', index=False)


# def logloss(expected, predicted):
#     logloss=−1N∑i=1N∑j=1Myijlog(pij),

prepare_solution()
