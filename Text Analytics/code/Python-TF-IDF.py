"""
This code is part of Data Science Dojo's bootcamp
Copyright (C) 2016

Objective: Building a TF-IDF matrix, including stemming for dictionary compression
Data Source: NLTK sample "movie_reviews" dataset
Python Version: 3.4+
Packages: nltk, scikit-learn, scipy, numpy
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import find
from scipy.spatial.distance import cosine
import numpy as np
import nltk 

# Download the nltk modules you need. You only need to run these lines once.
nltk.download('movie_reviews')
nltk.download('punkt')

from nltk.corpus import movie_reviews

# Description of data set
print(movie_reviews.readme())

# Prepare document set for stemming
movie_docs = [' '.join(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids()]

# Define function for tokenizing documents
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        item_lower = item.lower()
        stems.append(nltk.PorterStemmer().stem(item_lower))
    return stems

# Build TF-IDF matrix
tfidf = TfidfVectorizer(tokenizer=tokenize)
movie_tfidf = tfidf.fit_transform(movie_docs)

# Examine non-zero entries in TF-IDF matrix
find(movie_tfidf[1:6,1:10])

# Calculate pair-wise cosine similarities between first 200 documents
movie_cosine = np.empty((200, 200))
for ii in range(0, movie_cosine.shape[0]):
    for jj in range(ii, movie_cosine.shape[0]):
        movie_cosine[ii, jj] = cosine(movie_tfidf.getcol(ii).toarray(), movie_tfidf.getcol(jj).toarray())
        movie_cosine[jj, ii] = movie_cosine[ii, jj]

print(movie_cosine[1:6,1:6])