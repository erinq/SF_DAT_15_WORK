# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:50:14 2015

@author: equealy
"""
# lists

### tuple ###
'''
A tuple is a sequence of immutable Python objects. 
Tuples are sequences, just like lists. 
The differences between tuples and lists are,
the tuples cannot be changed unlike lists and 
tuples use parentheses, whereas lists use square brackets.
'''
#Python Expression	Results	                 Description
len((1, 2, 3))	      #3	                       Length
(1, 2, 3) + (4, 5, 6)	#(1, 2, 3, 4, 5, 6)	      Concatenation
('Hi!',) * 4	     #('Hi!', 'Hi!', 'Hi!', 'Hi!')	 Repetition
3 in (1, 2, 3)	     #True	                       Membership
for x in (1, 2, 3): print x,	#1 2 3	            Iteration:

# tuple filtering
L = ('spam', 'Spam', 'SPAM!')
L[2]	#'SPAM!'	Offsets start at zero
L[-2]	#'Spam'	Negative: count from the right
L[1:]	#['Spam', 'SPAM!']	Slicing fetches sections   
# functions
# cmp(tuple1, tuple 2) compare elements
# len(tuple) length funciton
# max(tuple)
# min(tuple)
# tuple(sequence) # turns a list into a tuple


###    pandas: data frame or series data ###
import pandas as pd
df = pd.read_csv('data/police-killings.csv')
df.columns # column names
df.head(3) # first 3 entries
df.shape # returns row, column
df.shape[0] # row
df.shape[1] # column
df.isnull().sum() #counts null values in frame
df.fillna(value='Unknown', inplace=True) # fills nulls with unknown, inplace changes frame
df[df.year == 2015] # filters only year 2015, returns data frame

# numpy array for numerical manipulation of numbers

# scikit learn
