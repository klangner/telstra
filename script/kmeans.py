#
# Bayes solution
#

from sklearn.cluster import KMeans
from datasets import load_cross_validation, Dataset
import numpy as np
import pandas as pd


def to_labels(labels):
    return pd.get_dummies(labels)


def check_score(expected, predicted):
    predicted = np.minimum(predicted, 1-10**-15)
    predicted = np.maximum(predicted, 10**-15)
    ps = np.multiply(expected, np.log(predicted))
    return -ps.sum().sum()/len(expected)

def cross_validate():
    print('Cross validate bayes model')
    train, test = load_cross_validation()
    X = train.get_features()
    Y = train.get_labels()
    X2 = test.get_features()
    Y2 = test.get_labels()
    kmeans = KMeans(n_clusters=8)
    clf = kmeans.fit(X, train.get_multi_labels())
    score = check_score(Y, to_labels(clf.predict(X)))
    print("Train dataset score %f" % score)
    score = check_score(Y2, to_labels(clf.predict(X2)))
    print("test dataset score %f" % score)


def save_predictions(y, test):
    predictions = pd.DataFrame(test['id'])
    predictions['id'] = test['id']
    predictions['predict_0'] = y[0]
    predictions['predict_1'] = y[1]
    predictions['predict_2'] = y[2]
    predictions[['id', 'predict_0', 'predict_1', 'predict_2']].to_csv('../data/solution.csv', index=False)


def submission():
    print('Cross validate K-Means model')
    train = Dataset.from_train()
    test = Dataset.from_test()
    X = train.get_features()
    Y = train.get_labels()
    X2 = test.get_features()
    kmeans = KMeans(n_clusters=8)
    clf = kmeans.fit(X, train.get_multi_labels())
    score = check_score(Y, to_labels(clf.predict(X)))
    print("Train dataset score %f" % (score/len(X)))
    Y2 = to_labels(clf.predict(X2))
    save_predictions(Y2, test.df)

if __name__ == "__main__":
    cross_validate()
    # submission()
