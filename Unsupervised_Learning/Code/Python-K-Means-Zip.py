"""
This code is part of Data Science Dojo's bootcamp
Copyright (C) 2016

Objective: Unsupervized learning of Zip dataset using K-means
Data Source: bootcamp root/Datasets/titanic.csv
Python Version: 3.4+
Packages: scikit-learn, pandas, matplotlib, numpy
"""
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in the data. Remember to set your working directory!
zip_train = pd.read_csv('Datasets/Zip/zip.train.csv', header=None)

# Build model
## Randomly sample 500 rows in training set
np.random.seed(27)
zip_cluster_indices = np.random.choice(len(zip_train), 500, replace=False)
zip_cluster = zip_train.iloc[zip_cluster_indices,]

## Run the clustering model
zip_km_model = KMeans(n_clusters=10, max_iter=300, n_init=10,
                        init='k-means++', tol=.0001, n_jobs=1)
zip_km_model = zip_km_model.fit(zip_cluster.iloc[:,1:])

# Visualize clusters
## Subset out each cluster to investigate data
zip_cluster.insert(1, 'assign', zip_km_model.labels_)
zip_grouped = zip_cluster.groupby('assign')
for name, group in zip_grouped:
    print(str(name) + ": " + str(group.iloc[:,0].values))

# Build skree plot.
zip_skree_inertia = []
clusters = range(2,16)
for n_cluster in clusters:
    km_model = KMeans(n_clusters=n_cluster, max_iter=300, n_init=10,
                      init='k-means++', tol=.0001, n_jobs=1
                      ).fit(zip_cluster.iloc[:,1:])
    zip_skree_inertia.append(km_model.inertia_)

plt.plot(clusters,zip_skree_inertia, 'bo')
plt.show()

## Exercise:
## Based on the scree plot, how many clusters would you choose to use if you didn't know there
## were 10 classes in this data?
## Play with different numbers of clusters and different initial conditions (try init='random')
## What different numbers are often clustered together? What does this indicate about the weaknesses
## and strengths of kmeans?