"""
This code is part of Data Science Dojo's bootcamp
Copyright (C) 2016

Objective: Machine Learning of the Kyphosis dataset with a Decision Tree
Data Source: bootcamp root/Datasets/kyphosis.csv
Python Version: 3.4+
Packages: scikit-learn, pandas, numpy, pydotplus
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, tree
from sklearn.externals.six import StringIO
import pydotplus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data. Remember to set your working directory to the bootcamp root.
kyphosis = pd.read_csv('Datasets/kyphosis.csv', index_col=0)
kyphosis["Kyphosis"] = pd.Categorical(kyphosis["Kyphosis"],
                                      categories=['absent', 'present'])

# Data exploration and visualization
kyphosis.boxplot(by='Kyphosis')
pd.tools.plotting.scatter_matrix(kyphosis.iloc[:,1:])
plt.show()

# Randomly choose 60% of the data as training data (Why 60% instead of 70%?)
np.random.seed(27)
kyphosis.is_train = np.random.uniform(0, 1, len(kyphosis)) <= .6
kyphosis_train = kyphosis[kyphosis.is_train]
kyphosis_test = kyphosis[kyphosis.is_train == False]

# Train model
kyphosis_features = kyphosis.columns[1:]
kyphosis_dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=None,
                                         min_samples_split=2, min_samples_leaf=1)
kyphosis_dt_clf = kyphosis_dt_clf.fit(kyphosis_train[kyphosis_features],
                                        kyphosis_train['Kyphosis'])

# Print a string representation of the tree.
# If you have graphviz (www.graphviz.org) installed, you can write a pdf
# visualization using graph.write_pdf(filename)
kyphosis_dt_data = StringIO()
tree.export_graphviz(kyphosis_dt_clf, out_file=kyphosis_dt_data)
kyphosis_dt_graph = pydotplus.parser.parse_dot_data(kyphosis_dt_data.getvalue())
print(kyphosis_dt_graph.to_string())

# Predict classes of test set and evaluate
kyphosis_dt_pred = kyphosis_dt_clf.predict(kyphosis_test[kyphosis_features])

kyphosis_dt_cm = metrics.confusion_matrix(kyphosis_test['Kyphosis'],
                                          kyphosis_dt_pred,
                                          labels=['absent', 'present'])
print(kyphosis_dt_cm)
kyphosis_dt_acc = metrics.accuracy_score(kyphosis_test['Kyphosis'],
                                         kyphosis_dt_pred)
kyphosis_dt_prec = metrics.precision_score(kyphosis_test['Kyphosis'],
                                           kyphosis_dt_pred,
                                           pos_label='absent')
kyphosis_dt_rec = metrics.recall_score(kyphosis_test['Kyphosis'],
                                       kyphosis_dt_pred,
                                       pos_label='absent')
kyphosis_dt_f1 = metrics.f1_score(kyphosis_test['Kyphosis'], kyphosis_dt_pred,
                                  pos_label='absent')
print("accuracy: " + str(kyphosis_dt_acc) + "\n precision: " 
      + str(kyphosis_dt_prec) + "\n recall: " + str(kyphosis_dt_rec)
      + "\n f1-score: " + str(kyphosis_dt_f1))