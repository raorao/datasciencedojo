###################################################################################
## This code is part of Data Science Dojo's bootcamp
## Copyright (C) 2015

## Objective: Building TF-IDF matrices using R, including stemming for dictionary compression
## Packages: tm, lsa
###################################################################################

# Load the library
library(tm)
library(lsa)

# Load the test dataset
data(crude)

# Explore the dataset
summary(crude)

# Remove punctuation, apply a stemmer, and build a document-term matrix using TF-IDF
crude.dt <- DocumentTermMatrix(crude, control=list(weighting=weightTfIdf,
                                                   removePunctuation=T,
                                                   stemming=T))

# Inspect the document-term matrix
print(crude.dt)
#inspect(crude.dt)

#Compute a matrix of cosine similarity scores between each document pair
crude.cos <- cosine(as.matrix(t(crude.dt)))
print(crude.cos)
