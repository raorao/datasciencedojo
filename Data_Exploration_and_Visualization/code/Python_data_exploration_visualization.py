"""
This code is part of Data Science Dojo's bootcamp
Copyright (C) 2016

Objective: Explore and visualize data using Python
Data Source: Multiple
Python Version: 3.4+
Packages: matplotlib, pandas, seaborn 
"""
# Script for following along in Data Exploration and Visualization module
# Set your working directory to the bootcamp root with os.chdir()
# Copy-paste line by line into an iPython shell
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import iris data into Pandas DataFrame
iris = pd.read_csv('Datasets/Iris_Data.csv')
iris.head()

# Pandas boxplot
iris.boxplot(column='Sepal.Length', by='Species', return_type='axes')

# Pandas boxplots with notches
plt.figure()
iris_box = iris.boxplot(column='Sepal.Length', by='Species', notch=True, return_type='axes')

# Saving Plots.
# Saves to current working directory (os.cwd()) by default
plt.figure()
iris_box['Sepal.Length'].get_figure().savefig("myplot.pdf", format='pdf')

# Matplotlib histogram
plt.figure()
plt.hist(iris['Petal.Width'], bins=20)

# Pandas density plot
plt.figure()
iris['Petal.Length'].plot(kind='kde')

# Seaborn multiple density plots
fig = plt.figure()
ax = fig.add_subplot(111)
for name, group in iris['Petal.Width'].groupby(iris['Species']):
    sns.distplot(group, hist=False, rug=True, ax=ax, label=name)
ax.legend()

# Exercise 1:
# Make a 2-D scatter plot of Sepal Length vs Sepal Width and 
# Petal Length vs Petal Width using core. Then recreate the same graphs in 
# lattice, this time coloring the individual points by species.

# Matplotlib scatter plot
plt.figure()
plt.scatter(x=iris['Sepal.Length'], y=iris['Sepal.Width'])

# Pandas segmented scatter plot
plt.figure()
iris_groups = iris.groupby('Species')
ax = iris_groups.get_group('setosa').plot(kind='scatter', x='Sepal.Length', y='Sepal.Width',
                                          label='Setosa', color='Blue')
iris_groups.get_group('virginica').plot(kind='scatter', x='Sepal.Length', y='Sepal.Width',
                                          label='Virginica', color='Green', ax=ax)
iris_groups.get_group('versicolor').plot(kind='scatter', x='Sepal.Length', y='Sepal.Width',
                                          label='Versicolor', color='Red', ax=ax)

# Seaborn regression lines
## Ungrouped
plt.figure()
sns.lmplot(x='Petal.Length', y='Petal.Width', data=iris)
## Grouped
plt.figure()
sns.lmplot(x='Petal.Length', y='Petal.Width', data=iris, hue='Species')

# Seaborn scatter plot matrix
plt.figure()
sns.pairplot(data=iris, hue='Species', diag_kind='kde', palette='husl', markers=['o', 's', 'D'])


#### Extended Titanic Exploration ####
# Read in the data and check structure
titanic = pd.read_csv('Datasets/titanic.csv')
titanic.head()

titanic.info()

# Casting & Readability
titanic['Survived'] = titanic['Survived'].astype('category')
titanic['Survived'].cat.categories = ['Dead', 'Survived']
titanic['Survived'].value_counts()
titanic['Embarked'] = titanic['Embarked'].astype('category')
titanic['Embarked'].cat.categories = ['Cherbourg', 'Queenstown', 'Southampton']
titanic['Embarked'].value_counts()
titanic.loc[:,['Survived', 'Embarked']].describe()

# Pie Chart
plt.figure()
titanic['Survived'].value_counts().plot(kind='pie')

# Is Sex a good predictor?
male = titanic[titanic['Sex']=='male']
female = titanic[titanic['Sex']=='female']
sex_values = pd.DataFrame({'male':male['Survived'].value_counts(),
                           'female':female['Survived'].value_counts()})
plt.figure()
sex_values.plot(kind='pie', subplots=True, figsize=(8,4))

# Is Age a good predictor?
titanic['Age'].describe()
titanic[titanic['Survived']=='Dead']['Age'].describe()
titanic[titanic['Survived']=='Survived']['Age'].describe()

# Exercise 3:
# Create 2 box plots of Age, one segmented by Sex, the other by Survived
# Create a histogram of Age
# Create 2 density plot of Age, also segmented by Sex and Survived
plt.figure()
titanic.boxplot(by='Sex', column='Age')
plt.figure()
titanic.boxplot(by='Survived', column='Age')

plt.figure()
titanic['Age'].hist(bins=12)

fig = plt.figure()
ax_sex = fig.add_subplot(211)
ax_surv = fig.add_subplot(212)
for name, group in titanic['Age'].groupby(titanic['Sex']):
    sns.distplot(group, hist=False, label=name, ax=ax_sex)
    
for name, group in titanic['Age'].groupby(titanic['Survived']):
    sns.distplot(group, hist=False, label=name, ax=ax_surv)
ax_sex.legend()
ax_surv.legend()

# Exercise 4:
# Create a new column "Child", and assign each row either "Adult" or "Child"
# based on a consistent metric. Then create a series of box plots
# relating Fare, Child, Sex, and Survived
child = titanic['Age']
child.loc[child < 13] = 0
child.loc[child >= 13] = 1
titanic['Child'] = child.astype('category')
titanic['Child'].cat.categories = ['Child', 'Adult']
# Run these next two lines as one block
plt.figure()
child_facets = sns.FacetGrid(data=titanic, row='Sex', col='Survived', sharey=True)
child_facets = child_facets.map(sns.boxplot, "Fare", "Child")
