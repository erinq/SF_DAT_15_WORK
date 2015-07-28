# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:54:59 2015

@author: equealy

exploration of Kaggle Crime Statistics Data Set
"""
import pandas as pd

# here is the test data - run model on this data and create output in the form of 
# the sample submission
#test = pd.read_csv('../data/kaggle-crime-test.csv')
test = pd.read_csv('https://dl.dropboxusercontent.com/u/4484471/kaggle-crime-test.csv')
test.shape
test.columns # fields: 'Id','Dates','DayOfWeek','PdDistrict','Address','X','Y'
             # where X = longitude
             # and   Y = latitude

# sample data
#sample = pd.read_csv('../kaggle-crime-sampleSubmission.csv')
sample = pd.read_csv('https://dl.dropboxusercontent.com/u/4484471/kaggle-crime-sampleSubmission.csv')
sample.columns
# this shows that based on the input data, we generate a prediction for
# the crime being in one of 39 categories:
# u'Id', u'ARSON', u'ASSAULT', u'BAD CHECKS', u'BRIBERY', u'BURGLARY', 
# u'DISORDERLY CONDUCT', u'DRIVING UNDER THE INFLUENCE', u'DRUG/NARCOTIC', 
# u'DRUNKENNESS', u'EMBEZZLEMENT', u'EXTORTION', u'FAMILY OFFENSES', 
# u'FORGERY/COUNTERFEITING', u'FRAUD', u'GAMBLING', u'KIDNAPPING', 
# u'LARCENY/THEFT', u'LIQUOR LAWS', u'LOITERING', u'MISSING PERSON', 
# u'NON-CRIMINAL', u'OTHER OFFENSES', u'PORNOGRAPHY/OBSCENE MAT', u'PROSTITUTION',
# u'RECOVERED VEHICLE', u'ROBBERY', u'RUNAWAY', u'SECONDARY CODES', 
# u'SEX OFFENSES FORCIBLE', u'SEX OFFENSES NON FORCIBLE', u'STOLEN PROPERTY',  
# u'SUICIDE', u'SUSPICIOUS OCC', u'TREA', u'TRESPASS', u'VANDALISM', 
# u'VEHICLE THEFT', u'WARRANTS', u'WEAPON LAWS'

# here is the training data - use this to make our model
#train = pd.read_csv('../data/kaggle-crime-train.csv')
train = pd.read_csv('https://dl.dropboxusercontent.com/u/4484471/kaggle-crime-train.csv')
# the training data 
# same as text data but also includes columns: Category and Resolution
# we are trying to predict Category

#for classification we can do
# knn
# kmeans - well we could  explore, and see if the crimes naturally separate 
#into 39 blobs, and then define the distances for those blobs.
# naive bayes
# I'm thinking about how to handle the 39 different classifications, as
# opposed to the binary classifications we reviewed in class
