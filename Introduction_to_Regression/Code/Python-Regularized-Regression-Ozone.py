"""
This code is part of Data Science Dojo's bootcamp
Copyright (C) 2016

Objective: Machine Learning of the Ozone dataset with linear regression and regularization
Data Source: bootcamp root/Datasets/titanic.csv
Python Version: 3.4+
Packages: scikit-learn, pandas, numpy
"""
from math import sqrt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
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

# Train Models to compare
## No Regularization
ozone_ln_reg = LinearRegression(normalize=True)
ozone_ln_reg = ozone_ln_reg.fit(ozone_train.drop('ozone', axis=1),ozone_train['ozone'])

## L2 (Ridge) Regularization
ozone_ridge_reg = Ridge(alpha=1.0, max_iter=1000, normalize=True, solver='lsqr',
                        tol=0.001)
ozone_ridge_reg = ozone_ridge_reg.fit(ozone_train.drop('ozone', axis=1),
                                      ozone_train['ozone'])

## L1 (Lasso) Regularization
ozone_lasso_reg = Lasso(alpha=1.0, max_iter=1000, normalize=True, tol=0.0001)
ozone_lasso_reg = ozone_lasso_reg.fit(ozone_train.drop('ozone', axis=1),
                                      ozone_train['ozone'])

## Cross validation to determined optimal alpha
ozone_ridgecv_reg = RidgeCV(alphas=(0.1, 1.0, 10.0), normalize=True, 
                            scoring='mean_absolute_error', cv=10)
ozone_ridgecv_reg = ozone_ridgecv_reg.fit(ozone_train.drop('ozone', axis=1),
                                          ozone_train['ozone'])

## Compare regularization models
print("Linear Coef: " + str(ozone_ln_reg.coef_)
      + "\nRidge Coef: " + str(ozone_ridge_reg.coef_) 
      + "\nLasso Coef: " + str(ozone_lasso_reg.coef_)
      + "\nCV Coef: " + str(ozone_ridgecv_reg.coef_)
      + "\nCV alpha: " + str(ozone_ridgecv_reg.alpha_))

# Predict using models and evaluate
ozone_ln_pred = ozone_ln_reg.predict(ozone_test.drop('ozone', axis=1))
ozone_ridge_pred = ozone_ridge_reg.predict(ozone_test.drop('ozone', axis=1))
ozone_lasso_pred = ozone_lasso_reg.predict(ozone_test.drop('ozone', axis=1))
ozone_ridgecv_pred = ozone_ridgecv_reg.predict(ozone_test.drop('ozone', axis=1))

## Calculate MAE, RMSE, and R-squared for all models
ozone_ln_mae = metrics.mean_absolute_error(ozone_test['ozone'], ozone_ln_pred)
ozone_ln_rmse = sqrt(metrics.mean_squared_error(ozone_test['ozone'], ozone_ln_pred))
ozone_ln_r2 = metrics.r2_score(ozone_test['ozone'], ozone_ln_pred)

ozone_ridge_mae = metrics.mean_absolute_error(ozone_test['ozone'], ozone_ridge_pred)
ozone_ridge_rmse = sqrt(metrics.mean_squared_error(ozone_test['ozone'], ozone_ridge_pred))
ozone_ridge_r2 = metrics.r2_score(ozone_test['ozone'], ozone_ridge_pred)

ozone_lasso_mae = metrics.mean_absolute_error(ozone_test['ozone'], ozone_lasso_pred)
ozone_lasso_rmse = sqrt(metrics.mean_squared_error(ozone_test['ozone'], ozone_lasso_pred))
ozone_lasso_r2 = metrics.r2_score(ozone_test['ozone'], ozone_lasso_pred)

ozone_ridgecv_mae = metrics.mean_absolute_error(ozone_test['ozone'], ozone_ridgecv_pred)
ozone_ridgecv_rmse = sqrt(metrics.mean_squared_error(ozone_test['ozone'], ozone_ridgecv_pred))
ozone_ridgecv_r2 = metrics.r2_score(ozone_test['ozone'], ozone_ridgecv_pred)

ozone_eval = pd.DataFrame({"Linear":{"MAE":ozone_ln_mae, "RMSE":ozone_ln_rmse, "R2":ozone_ln_r2},
                           "Ridge":{"MAE":ozone_ridge_mae, "RMSE":ozone_ridge_rmse,
                                    "R2":ozone_ridge_r2},
                           "Lasso":{"MAE":ozone_lasso_mae, "RMSE":ozone_lasso_mae,
                                    "R2":ozone_lasso_r2},
                           "Ridge CV":{"MAE":ozone_ridgecv_mae, "RMSE":ozone_ridgecv_rmse,
                                       "R2":ozone_ridgecv_r2}
                          })
print(ozone_eval)
