#
# Main function which calculates score on cross_validation and prepares solution.
#
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datasets import Dataset, save_predictions, load_cross_validation


def prepare_solution():
    train = Dataset.from_train()
    X = train.get_features()
    Y = train.get_labels()
    rf = RandomForestRegressor(n_jobs=-1)
    model = rf.fit(X, Y)
    print('Train score: %f' % loss(Y, model.predict(X)))
    test = Dataset.from_test()
    X2 = test.get_features()
    Y2 = model.predict(X2)
    save_predictions(Y2, test)


def cross_validate():
    train, test = load_cross_validation()
    X = train.get_features()
    Y = train.get_labels()
    X2 = test.get_features()
    Y2 = test.get_labels()
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
