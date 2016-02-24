"""
This code is part of Data Science Dojo's bootcamp
Copyright (C) 2016

Objective: Machine Learning of the Iris dataset with a Decision Tree
Data Source: scikit-learn iris dataset
Python Version: 3.4+
Packages: scikit-learn, pandas, numpy, pydotplus
"""
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, tree
from sklearn.externals.six import StringIO
import pydotplus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the iris dataset into memory
iris = load_iris()

# Create a Pandas dataframe from the sklearn dataset for visualization ease
irisDF = pd.DataFrame(iris.data, columns=iris.feature_names)
irisDF['Species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Data Exploration and Visualization
print(iris.DESCR)
irisDF.boxplot(by='Species')
pd.tools.plotting.scatter_matrix(irisDF.iloc[:,:4])
plt.show()

# Split Data for training and test
np.random.seed(27)
irisDF.is_train = np.random.uniform(0, 1, len(irisDF)) <= .7
irisDF_train = irisDF[irisDF.is_train]
irisDF_test = irisDF[irisDF.is_train == False]

# Train model
iris_features = irisDF.columns[:4]
iris_dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=None, 
                                     min_samples_split=2, min_samples_leaf=1)
iris_dt_clf = iris_dt_clf.fit(irisDF_train[iris_features], 
                              irisDF_train['Species'])
# Print a string representation of the tree.
# If you have graphviz (www.graphviz.org) installed, you can write a pdf
# visualization using graph.write_pdf(filename)
iris_dt_data = StringIO()
tree.export_graphviz(iris_dt_clf, out_file=iris_dt_data)
iris_dt_graph = pydotplus.parser.parse_dot_data(iris_dt_data.getvalue())
print(iris_dt_graph.to_string())

# Predict classes of test set and evaluate
iris_dt_pred = iris_dt_clf.predict(irisDF_test[iris_features])

iris_dt_cm = metrics.confusion_matrix(irisDF_test['Species'], iris_dt_pred,
                                          labels=('setosa','versicolor',
                                                  'virginica')
                                     )
print(iris_dt_cm)
iris_dt_acc = metrics.accuracy_score(irisDF_test['Species'], iris_dt_pred)
iris_dt_prec = metrics.precision_score(irisDF_test['Species'], iris_dt_pred,
                                       average='weighted')
iris_dt_rec = metrics.recall_score(irisDF_test['Species'], iris_dt_pred,
                                   average='weighted')
iris_dt_f1 = metrics.f1_score(irisDF_test['Species'], iris_dt_pred,
                              average='weighted')
print("accuracy: " + str(iris_dt_acc) + "\n precision: " 
      + str(iris_dt_prec) + "\n recall: " + str(iris_dt_rec)
      + "\n f1-score: " + str(iris_dt_f1))