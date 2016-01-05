#
# Explore dataset
#

from datasets import Dataset


def main():
    print('Explore dataset')
    train = Dataset.from_train()
    u = train.pca()
    print('U shape: ' + str(u.shape))
    X = train.get_pca_features(u)
    print(X.shape)

if __name__ == "__main__":
    main()
