"""
This code is part of Data Science Dojo's bootcamp
Copyright (C) 2016

Objective: Machine Learning of the Zip dataset with logistic regression. Binary classification of '2' and '3'.
Data Source: bootcamp root/Datasets/titanic.csv
Python Version: 3.4+
Packages: scikit-learn, pandas, matplotlib
"""
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

# Read in the Data. Remember to set your working directory!
zip_train = pd.read_csv('Datasets/Zip/zip.train.csv', header=None)
zip_test = pd.read_csv('Datasets/Zip/zip.test.csv', header=None)


# Data Exploration and Visualization
## Check the dimensions of the data
print(zip_train.shape)
print(zip_test.shape)

## Check the first few rows
## Column 1 is the number label, 2 -> 257 are pixel grey levels
print(zip_train.head())

## Label target column as 'target'
zip_train.rename(columns={0:'target'}, inplace=True)
zip_test.rename(columns={0:'target'}, inplace=True)
## Visualize a number
zip_example = zip_train.iloc[4,1:].values
zip_example.shape = (16,16)
plt.imshow(zip_example, interpolation='none')
plt.show()

# Train model
## Filter out the "2" and "3" labels from the training and test sets
zip_train = zip_train[(zip_train['target'] == 2) | (zip_train['target'] == 3)]
zip_test = zip_test[(zip_test['target'] == 2) | (zip_test['target'] == 3)]

## Build model
zip_log_reg = LogisticRegression(penalty='l1', C=0.1, class_weight='balanced',
                                 max_iter=100, solver='liblinear', tol=.0001,
                                 n_jobs=1, random_state=27)
zip_log_reg = zip_log_reg.fit(zip_train.iloc[:,1:], zip_train['target'])

# Predict on test set for evaluation
zip_log_pred = zip_log_reg.predict(zip_test.iloc[:,1:])

zip_log_cm = metrics.confusion_matrix(zip_test['target'], zip_log_pred)
print(zip_log_cm)

zip_log_acc = metrics.accuracy_score(zip_test['target'], zip_log_pred)
zip_log_prec = metrics.precision_score(zip_test['target'], zip_log_pred, pos_label=3)
zip_log_rec = metrics.recall_score(zip_test['target'], zip_log_pred, pos_label=3)
zip_log_f1 = metrics.f1_score(zip_test['target'], zip_log_pred, pos_label=3)

print("Accuracy: " + str(zip_log_acc) + "\nPrecision: " + str(zip_log_prec)
      + "\nRecall: " + str(zip_log_rec) + "\nF1 score: " + str(zip_log_f1))

## Predict probabilities to get ROC AUC score
zip_log_prob = zip_log_reg.predict_proba(zip_test.iloc[:,1:])

zip_log_auc = metrics.roc_auc_score(zip_test['target'] - 2, zip_log_prob[:,1])

print('AUC: ' + str(zip_log_auc))