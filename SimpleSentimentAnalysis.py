
# coding: utf-8

# In[141]:


'''This is a script to predic polarity from a given set of textual data. We will be using three algorithms: logistic regression, 
support vector machine (with a linear kernel) and Naive Bayes algorithm. We will also be using a lot of different preprocessing
techniques.'''

import nltk
import sklearn
import numpy as np


f1 = open('rt-polarity.pos')
# The file path given above is the one in my disk system. Please change the path if you test it in your system
s = ''
for f in f1:
    s = s + f

# Initializing a list of positive reviews    
pos_list = s.split('\n')
pos_targ = [1] * len(pos_list)

f2 = open('rt-polarity.neg')
s = ''
for f in f2:
    s = s + f

# Initializing a list of negative reviews
neg_list = s.split('\n')
neg_targ = [0] * len(neg_list)


# In[142]:


data = pos_list + neg_list
targ = pos_targ + neg_targ

# Now we will construct a list of stop words from the data. For this, we will use NLTK's stopwords
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pprint import pprint
from time import time
import logging
stop_words = set(stopwords.words('english')) #This is the list of stop words

# The next step would be to construct a bag-of-words 
'''
Since creatung a bag-of-words is a part of pre-processing the textual data, we will be choosing some parameters and observing
the results:
1. Experimenting with the complexity of N-gram features (unigram, or unigram and bigram).
2. Removing/not removing stop words.
3. Remove infrequent words and choose the threshold at which to remove it.
'''
# There is also some tuning that should be done while training the models like experimenting with the amount of 
# smoothing/regularizationin training the methods

# To make the code more compact and easily find the parameters that produce the best results, we will be using a GridSearchCV pipleine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# First we should split the complete dataset and labels into training and test sets
text_train, text_test, targ_train, targ_test = train_test_split(data, targ, test_size = 0.2, random_state = 0, shuffle = True)

# Now let us construct a GridSearch pipeline with a text feature extractor and linear SVM classifier
# Adding TFIDF is optional, but preferable as the word counts are normalized
pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LinearSVC())])

# Let us also define the range of the different paramaters
parameters = {'vect__min_df' : (0.01, 0.02, 0.05),
             'vect__ngram_range' : ((1, 1), (1, 2)),   # unigram or bigram
             'vect__stop_words' : (None, stop_words),   # list of stop words from NLTK with an option of not having stop words
             'vect__decode_error': ['replace'],
              'tfidf__use_idf': (True, False),
              'tfidf__norm': ('l1', 'l2'),              
             'clf__penalty': ['l1', 'l2'],
              'clf__dual': [False]
             }

grid_search = GridSearchCV(pipeline, parameters)
print 'Performing Grid Search for SVM'
print 'pipeline:', [name for name, _ in pipeline.steps]
print 'parameters'
pprint(parameters)
t0 = time()
grid_search.fit(text_train, targ_train)
print 'done in %0.3fs' % (time()-t0)
print ''

print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set: '
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s:   %r' % (param_name, best_parameters[param_name])


# In[143]:


targ_pred = grid_search.predict(text_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print 'Accuracy score:', accuracy_score(targ_test, targ_pred)
print 'Confusion matrix: \n'
print confusion_matrix(targ_test, targ_pred)


# In[144]:


# Now we will compare the model's prediction with a baseline random predictor
import random
targ_rand = [random.randint(0, 1) for _ in range(len(targ_test))]
print 'Accuracy score of a random predictor:', accuracy_score(targ_test, targ_rand)


# In[145]:


# Now we repeat the same process for a Logistic Regression classifier
pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression())])
parameters = {'vect__min_df' : (0.01, 0.02, 0.05),
             'vect__ngram_range' : ((1, 1), (1, 2)),   # unigram or bigram
             'vect__stop_words' : (None, stop_words),   # list of stop words from NLTK with an option of not having stop words
             'vect__decode_error': ['replace'],
             'tfidf__use_idf': (True, False),
             'tfidf__norm': ('l1', 'l2'),    
             'clf__penalty': ['l1', 'l2'],
              'clf__dual': [False]
             }

grid_search = GridSearchCV(pipeline, parameters)
print 'Performing Grid Search for Logistic Regression'
print 'pipeline:', [name for name, _ in pipeline.steps]
print 'parameters'
pprint(parameters)
t0 = time()
grid_search.fit(text_train, targ_train)
print 'done in %0.3fs' % (time()-t0)
print ''

print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set: '
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s:   %r' % (param_name, best_parameters[param_name])


# In[146]:


targ_pred = grid_search.predict(text_test)
from sklearn.metrics import accuracy_score
print 'Accuracy score:', accuracy_score(targ_test, targ_pred)
print 'Confusion matrix: \n'
print confusion_matrix(targ_test, targ_pred)


# In[147]:


# Now we will compare the model's prediction with a baseline random predictor
import random
targ_rand = [random.randint(0, 1) for _ in range(len(targ_test))]
print 'Accuracy score of a random predictor:', accuracy_score(targ_test, targ_rand)


# In[148]:


# Finally we will perform the last experiment with a Naive Bayes classifier (Multinomial Naive Bayes)
pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
parameters = {'vect__min_df' : (0.01, 0.02, 0.05),
             'vect__ngram_range' : ((1, 1), (1, 2)),   # unigram or bigram
             'vect__stop_words' : (None, stop_words),   # list of stop words from NLTK with an option of not having stop words
             'vect__decode_error': ['replace'],
              'tfidf__use_idf': (True, False),
              'tfidf__norm': ('l1', 'l2'),    
             'clf__alpha': (0.1, 0.2, 0.5, 1)
             }
grid_search = GridSearchCV(pipeline, parameters)
print 'Performing Grid Search for Naive Bayes'
print 'pipeline:', [name for name, _ in pipeline.steps]
print 'parameters'
pprint(parameters)
t0 = time()
grid_search.fit(text_train, targ_train)
print 'done in %0.3fs' % (time()-t0)
print ''

print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set: '
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s:   %r' % (param_name, best_parameters[param_name])


# In[149]:


targ_pred = grid_search.predict(text_test)
from sklearn.metrics import accuracy_score
print 'Accuracy score:', accuracy_score(targ_test, targ_pred)
print 'Confusion matrix: \n'
print confusion_matrix(targ_test, targ_pred)


# In[150]:


# Now we will compare the model's prediction with a baseline random predictor
import random
targ_rand = [random.randint(0, 1) for _ in range(len(targ_test))]
print 'Accuracy score of a random predictor:', accuracy_score(targ_test, targ_rand)

