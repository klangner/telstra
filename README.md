# Telstra Kaggle competition:

https://www.kaggle.com/c/telstra-recruiting-network


## Results

### Score: 0.69940

  * 1 Hidden layer with 10 units
  * Learning rate: 0.0003
  * Softmax in output layer
  * Log features are normalized by biggest value in all records.
  * 37 hours of training.
  * Looks like overfitting. On training set the score is 0.5484

### Score: 0.63708

  * Log features are counted as boolean value (Column volume is not used)
  * Still overfits. Training stopped after score on training set: 0.557506
  * Training time: 6 minutes

## Features

Testing different combination of features with cross validation score

  * With all feature score: 0.6287 (train: 0.5297)
  * Log features only, score: 0.6497
  * Without 'severity_type', score: 0.6287 (train: 0.5363)
  * Without 'resource_type', score: 0.6430 (train: 0.5213)

## What works

  * ReLu units performs better then sigmoid
  * Log features without volume column


## What doesnt work

  * Switching to logistic units
  * Adding second hidden layer
  * More then 5 hidden units (in single layer) overfits network