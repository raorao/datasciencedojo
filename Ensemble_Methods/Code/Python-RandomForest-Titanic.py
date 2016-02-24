"""
This code is part of Data Science Dojo's bootcamp
Copyright (C) 2016

Objective: Machine Learning of the Titanic dataset with a Random Forest
Data Source: bootcamp root/Datasets/titanic.csv
Python Version: 3.4+
Packages: scikit-learn, pandas, numpy, sklearn-pandas
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
from sklearn_pandas import DataFrameMapper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data into a pandas data frame. Remember to set your working directory!
titanic = pd.read_csv('Datasets/titanic.csv')

# Data Exploration and Cleaning
## Remove PassengerID, Name, Ticket, and Cabin attributes
titanic = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'], axis=1)

## Set values to categorical and clean NaNs as needed
titanic['Survived'] = titanic['Survived'].astype('category')
titanic['Pclass'] = titanic['Pclass'].astype('category')
titanic.loc[pd.isnull(titanic['Embarked']), 'Embarked'] = 'U'
titanic['Embarked'] = titanic['Embarked'].astype('category')
titanic['Sex'] = titanic['Sex'].astype('category')

## Data Visualization
titanic.boxplot(by='Survived')
pd.tools.plotting.scatter_matrix(titanic)
titanic.describe()
titanic.loc[:,['Survived', 'Pclass', 'Embarked', 'Sex']].describe()
plt.show()

## Replace missing values in Age with mean and Embarked with NA
titanic.loc[pd.isnull(titanic['Age']), 'Age'] = titanic['Age'].mean()

## Encode all categorical values as integers
## Survived: 0 = Dead, 1 = Alive
## Embarked: 0 = Cherbourg, 1 = Queenstown, 2 = Southampton, 3 = Unknown
## Sex: 0 = female, 1 = male
titanic['Survived'].cat.categories = [0,1]
titanic['Embarked'].cat.categories = [0,1,2,3]
titanic['Sex'].cat.categories = [0,1]

## Create Data Frame to nparray mapping
titanic_map = DataFrameMapper([
    ('Age', None),
    ('SibSp', None),
    ('Parch', None),
    ('Sex', None),
    (['Embarked'], LabelBinarizer()),
    (['Pclass'], LabelBinarizer())
])
titanic_cln_var_names = ['Age', 'SibSp', 'Parch', 'Sex', 'Embarked-C', 'Embarked-Q',
                         'Embarked-S', 'Embarked-U', 'Pclass-1', 'Pclass-2', 'Pclass-3']

titanic_features = titanic_map.fit_transform(titanic)

# Split Data for training and test
np.random.seed(27)
titanic.is_train = np.random.uniform(0,1,len(titanic)) <= .7
titanic_features_train = titanic_features[titanic.is_train]
titanic_features_test = titanic_features[titanic.is_train == False]


# Train Model
titanic_rf_clf = RandomForestClassifier(oob_score=True, n_jobs=2, n_estimators=500, 
                                        max_features='sqrt', criterion='gini', max_depth=None, 
                                        min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None
                                       )
titanic_rf_clf = titanic_rf_clf.fit(titanic_features_train, 
                                    titanic.loc[titanic.is_train,'Survived'])
print("OOB Accuracy: " + str(titanic_rf_clf.oob_score))

# Predict classes of test set and evaluate
titanic_rf_pred = titanic_rf_clf.predict(titanic_features_test)

titanic_rf_cm = metrics.confusion_matrix(titanic.loc[titanic.is_train==False, 'Survived'],
                                         titanic_rf_pred, labels=[0,1])
print(titanic_rf_cm)
titanic_rf_acc = metrics.accuracy_score(titanic.loc[titanic.is_train==False, 'Survived'],
                                         titanic_rf_pred)
titanic_rf_prec = metrics.precision_score(titanic.loc[titanic.is_train==False, 'Survived'],
                                          titanic_rf_pred)
titanic_rf_rec = metrics.recall_score(titanic.loc[titanic.is_train==False, 'Survived'],
                                      titanic_rf_pred)
titanic_rf_f1 = metrics.f1_score(titanic.loc[titanic.is_train==False, 'Survived'],
                                      titanic_rf_pred)

# Predict probabilities to calculate AUC
titanic_rf_pred_prob = titanic_rf_clf.predict_proba(titanic_features_test)

titanic_rf_auc = metrics.roc_auc_score(titanic.loc[titanic.is_train==False, 'Survived'],
                                       titanic_rf_pred_prob[:,1])

print("Accuracy: " + str(titanic_rf_acc) + "\nPrecision: " 
      + str(titanic_rf_prec) + "\nRecall: " + str(titanic_rf_rec)
      + "\nF1-score: " + str(titanic_rf_f1) + "\nAUC: " + str(titanic_rf_auc))

# Feature Importance. Recall that we binarized the categorical columns, so there are 12 numbers.
# The variables are in the same order as in the titanic_map defintion above. 
# [Age, SibSp, Parch, Fare, Sex, Embarked-C, Embarked-Q, Embarked-S, Embarked-U, Pclass-1, Pclass-2,
#  Pclass-3]
# Larger numbers indicate a higher feature importance.
print(pd.Series(titanic_rf_clf.feature_importances_, index=titanic_cln_var_names).sort_values(ascending=False))
