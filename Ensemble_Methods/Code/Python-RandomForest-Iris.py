from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas
import numpy

# Load iris class into memory
iris = load_iris() 

# gaining information about the dataset, 
# would break in AzureML and AWS
#print iris['DESCR']

# loads the iris dataset into a data frame (pandas)
# inserts column names into the data frame
irisDF = pandas.DataFrame(iris.data, columns=iris.feature_names)

# Returns an array of 150 randomly true false values, 75% true, 25% false.
# The array index will be used to determine which index will belong to test
# set or train set. True will be train set, false will be test set.
irisDF['is_train'] = numpy.random.uniform(0, 1, len(irisDF)) <= .75

# specify species as the categorical value, then head to double check
irisDF['species'] = pandas.Categorical.from_codes(iris.target, iris.target_names)
irisDF.head()

# Filter out the true columns for the 75% partition
irisDF_train = irisDF[irisDF['is_train']==True]
# Filter out the non true columns for the 75% partition, aka, selecting only
# 25% partition.
irisDF_test = irisDF[irisDF['is_train']==False]

# Selecting predictors as the first 4 columns
features = irisDF.columns[:4]

# initiates the random forest algorithm
# n_jobs = 2 means to utilize 2 cores on the computer. If -1 then it'll be
# set to the number of cores on the computer.
# If the computer has hyper threading or dual threading, then 1.5x the number
# of cores as the maximum. More cores = faster training, but slower computer. 
# If this is running on your own computer, leave at least one core free 
# for the operating system and background functions if this operation is 
# going to take a long time.
# n_estimators = number of trees
# max_features = mtry
# More help
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
clf = RandomForestClassifier(n_jobs=2, n_estimators=500, max_features=2)
# Compresses categorical values into integer bins, speeds up training
y, _ = pandas.factorize(irisDF_train['species'])

# Trains the model/algorithm
clf.fit(irisDF_train[features], y)

# Scores/predicts all values for the test set using the trained model
preds = iris.target_names[clf.predict(irisDF_test[features])]
pandas.crosstab(
    irisDF_test['species'], 
    preds, 
    rownames=['actual'], 
    colnames=['preds']
)