"""
This code is part of Data Science Dojo's bootcamp
Copyright (C) 2016

Objective: Machine Learning of the Zip dataset with logistic regression. Binary classification of '2' and '3'.
Data Source: bootcamp root/Datasets/titanic.csv
Python Version: 3.4+
Packages: scikit-learn, pandas, numpy
"""
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pandas as pd
import numpy as np

# Read in the data. Remember to set your working directory!
spam = pd.read_csv('Datasets/spambase.data', header=None)

# Data exploration
## This data frame "spam" is the result of a text mining task based on 4601 emails.
## We want to classify an email as spam or regular email, which is recorded in column 57, where 1 = spam and 0 = not spam.
## The 1st to the 54th columns are the frequences of some important words or characters in each email.
## The 55th to 57th columns are some features reflecting the appearance of capital letters
## A more detailed explanation of these features is explained at the UCI repository
## url: https://archive.ics.uci.edu/ml/datasets/Spambase
print(spam.shape)
spam.describe()

## One attribute, column 40, has a perfect relationship with the spam/email label.
## As a result, if we leave the column in, the NaiveBayes function will error.
## Why is this good behavior?
spam.pop(40)

## Rename the target column as "spam" and cast it to categorical
spam.rename(columns={57:"spam"}, inplace=True)
spam['spam'] = spam['spam'].astype('category')
spam['spam'].describe()

# Train Model
## Split data into training and test sets
np.random.seed(27)
spam.is_train = np.random.uniform(0, 1, len(spam)) <= 0.7
spam_train = spam[spam.is_train]
spam_test = spam[spam.is_train == False]

## Check the class distributions of the training and test sets vs the whole set
spam_train['spam'].describe()
spam_test['spam'].describe()

## Build model
spam_nb_clf = GaussianNB()
spam_nb_clf = spam_nb_clf.fit(spam_train.drop('spam', axis=1), spam_train['spam'])

# Predict training classes and evaluate model
spam_nb_pred = spam_nb_clf.predict(spam_test.iloc[:,:56])

spam_nb_cm = metrics.confusion_matrix(spam_test['spam'], spam_nb_pred)
print(spam_nb_cm)

spam_nb_acc = metrics.accuracy_score(spam_test['spam'], spam_nb_pred)
spam_nb_prec = metrics.precision_score(spam_test['spam'], spam_nb_pred)
spam_nb_rec = metrics.recall_score(spam_test['spam'], spam_nb_pred)
spam_nb_f1 = metrics.f1_score(spam_test['spam'], spam_nb_pred)

# Predict training class probabilities and calculate ROC AUC score
spam_nb_prob = spam_nb_clf.predict_proba(spam_test.iloc[:,:56])

spam_nb_auc = metrics.roc_auc_score(spam_test['spam'], spam_nb_prob[:,1])

print("Accuracy: " + str(spam_nb_acc) + "\nPrecision: " 
      + str(spam_nb_prec) + "\nRecall: " + str(spam_nb_rec)
      + "\nF1-score: " + str(spam_nb_f1) + "\nAUC: " + str(spam_nb_auc))