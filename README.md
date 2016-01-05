# Telstra Kaggle competition:

https://www.kaggle.com/c/telstra-recruiting-network

## To try

  * Add regularization
  * Check how many PCA dimension to use to retain 99% of variability
  * Prepare confusion matrix


## Results

### Score: 0.69940

  * 1 Hidden layer with 10 units
  * Learning rate: 0.0003
  * Softmax in output layer
  * Log features are normalized by biggest value in all records.
  * 37 hours of training.
  * Looks like over fitting. On training set the score is 0.5484

### Score: 0.63708

  * Log features are counted as boolean value (Column volume is not used)
  * Still over fits. Training stopped after score on training set: 0.557506
  * Training time: 6 minutes

## Features

Testing different combination of features with cross validation score

  * With all feature score: 0.6287 (train: 0.5297)
  * Log features only, score: 0.6497
  * Without 'severity_type', score: 0.6287 (train: 0.5363)
  * Without 'resource_type', score: 0.6430 (train: 0.5213)

## Hyper-parameters

  * Number of hidden layers (1 and 2 tested)
  * Type of activation function (ReLu learn faster and over fits. sigmoid learn slower)
  * Hidden units from 3 to 100 (5-10 looks like optimal value)
  * Use of PCA to reduce dimension.

## What works

  * ReLu units performs better then sigmoid
  * Log features without volume column
  * PCA reduces features from 454 to 267


## What doesnt work

  * Switching to logistic units
  * Adding second hidden layer
  * More then 5 hidden units (in single layer) over fits network
  * Using PCA makes over fitting bigger problem