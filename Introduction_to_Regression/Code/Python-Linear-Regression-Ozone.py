"""
This code is part of Data Science Dojo's bootcamp
Copyright (C) 2016

Objective: Machine Learning of the Ozone dataset with linear regression
Data Source: bootcamp root/Datasets/titanic.csv
Python Version: 3.4+
Packages: scikit-learn, pandas, numpy
"""
from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read in the data. Remember to set your working directory!
ozone = pd.read_csv('Datasets/ozone.data', delimiter='\t')

# Data Visualization
ozone.describe()
pd.tools.plotting.scatter_matrix(ozone)
plt.show()

# Split data into training and test
np.random.seed(27)
ozone.is_train = np.random.uniform(0, 1, len(ozone)) <= .7
ozone_train = ozone[ozone.is_train]
ozone_test = ozone[ozone.is_train == False]

# Train Model
ozone_ln_reg = LinearRegression(normalize=True)
ozone_ln_reg = ozone_ln_reg.fit(ozone_train.drop('ozone', axis=1),ozone_train['ozone'])

print(ozone_ln_reg.coef_)

# Predict values of test set
ozone_ln_pred = ozone_ln_reg.predict(ozone_test.drop('ozone', axis=1))

# Evaluate model with visualization and numeric metrics
ozone_ln_resid = ozone_ln_pred - ozone_test['ozone']
plt.figure(2)
plt.scatter(ozone_test['ozone'], ozone_ln_resid)
plt.ylabel('Residuals')
plt.xlabel('True Values')
plt.figure(3)
plt.scatter(ozone_ln_pred, ozone_ln_resid)
plt.ylabel('Residuals')
plt.xlabel('Predicted Values')
plt.show()

ozone_ln_mae = metrics.mean_absolute_error(ozone_test['ozone'], ozone_ln_pred)
ozone_ln_rmse = sqrt(metrics.mean_squared_error(ozone_test['ozone'], ozone_ln_pred))
ozone_ln_r2 = metrics.r2_score(ozone_test['ozone'], ozone_ln_pred)

print("MAE: " + str(ozone_ln_mae) + "\nRMSE: " + str(ozone_ln_rmse) 
      + "\nCoefficient of Determination: " + str(ozone_ln_r2))
