#
# Explore dataset
#

from datasets import Dataset
from scipy import linalg
import numpy as np


def svd(df):
    U, s, Vh = linalg.svd(df)
    x = np.floor(np.minimum(s, 1))
    print(s)
    print(x.sum())


def main():
    print('Explore dataset')
    train = Dataset.from_train()
    X = train.get_features()
    Y = train.get_labels()
    svd(X)

if __name__ == "__main__":
    main()
