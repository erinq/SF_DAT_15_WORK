# by Erin Quealy for General Assembly Homework 2

##### Part 1 #####

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
import statsmodels.formula.api as smf

#import seaborn as sns
#import matplotlib.pyplot as plt

# 1. read in the yelp dataset

yelp = pd.read_csv('../data/yelp.csv') # if want to index by a different column
                                        # use index_col=column-number
yelp.head() # ??? difference head() and head
yelp.describe()
yelp.shape # ??? why is yelp.shape() not working
yelp.columns

# 2. Perform a linear regression using 
# "stars" as your response and 
# "cool", "useful", and "funny" as predictors

# this fits the line 
# stars = intercept + coefficient1(cool) + coefficent2(useful) + coefficent3(funny)

linreg = LinearRegression()
features = ['cool', 'useful', 'funny']
X = yelp[features]
y = yelp['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
# print linreg.intercept_ #y intercept
# print linreg.coef_ # slope of each predictor

coeffs = zip(features, linreg.coef_)
print 'Linear regression coeffiecients', coeffs

# 3. Show your MAE, R_Squared and RMSE

# model prediction: y_pred
y_pred = linreg.predict(X_test)
# mean absolute error MAE
print 'MAE single split ', metrics.mean_absolute_error(y_test, y_pred)
# mean squared error MSE
print 'MSE single split ', metrics.mean_squared_error(y_test, y_pred)
# R squared
print 'R^2 single split ', metrics.r2_score(y_test, y_pred)
# RMSE
print 'RMSE single split', np.sqrt(metrics.mean_squared_error(y_test, y_pred))

# add cross validation -
mae_scores = np.absolute(cross_val_score(linreg, X, y, scoring='mean_absolute_error', cv=5))
print 'cross-validated Mean Absolute Error average: ', mae_scores.mean()
mse_scores = np.absolute(cross_val_score(linreg, X, y, scoring='mean_squared_error', cv=5))
print 'cross-validated Mean Squared Error average: ', mse_scores.mean()
r2_scores = np.absolute(cross_val_score(linreg, X, y, scoring='r2', cv=5))
print 'cross-validated R-Squared average: ', r2_scores.mean()
print 'cross-validated Root Mean Squared Error average: ', np.sqrt(mse_scores.mean())


# 4. Use statsmodels to show your pvalues
# for each of the three predictors
# Using a .05 confidence level, 
# Should we eliminate any of the three?

# linear model is stats model formula ordinary least squares
lm = smf.ols(formula='stars ~ cool + useful + funny', data=yelp).fit()
lm.params
lm.pvalues # result is in predictive value cool>funny>useful 
            # useful is the least useful. hahaha
            # is this for a 95% confidence interval, how do we adjust confidence interval?
# all of the pvalues are less than .05, so all factors have a relationship to the response

# 5. Create a new column called "good_rating"
# this column should be True iff stars is 4 or 5
# and False iff stars is below 4
yelp['good_rating'] = yelp['stars']>=4

# 6. Perform a Logistic Regression using 
# "good_rating" as your response and the same
# three predictors
logreg = LogisticRegression() #instantiate
features = ['cool', 'useful', 'funny'] #predictors
X = yelp[features]
y = yelp['good_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg.fit(X_train, y_train)
zip(features, logreg.coef_[0]) # ??? why the [0]

# 7. Show your Accuracy, Sensitivity, Specificity
# and Confusion Matrix

# confusion matrix
y_pred = logreg.predict(X_test)
confusion = metrics.confusion_matrix(y_test, y_pred)
print confusion

# accuracy
print 'accuracy ', metrics.accuracy_score(y_test, y_pred)
#senstivity
print 'sensitivity ', 1.0*confusion[1][1] / (confusion[1][0] + confusion[1][1]) 
#specificity
print 'specificity ', 1.0*confusion[0][0] / (confusion[0][0] + confusion[0][1]) 
# cross validated accuracy
accuracy = np.absolute(cross_val_score(logreg, X, y, scoring='accuracy', cv=5))
print ' cross-validated accuracy ', accuracy.mean()

# 8. Perform one NEW operation of your 
# choosing to try to boost your metrics!

# adding a new predictor which is the product of useful and funny
# since they both negatively correlate with stars rating, maybe this will be 
# better at predicting bad reviews
yelp['new_predictor'] = yelp.useful * yelp.funny
logreg = LogisticRegression() #instantiate
features = ['cool', 'funny', 'useful', 'new_predictor'] #predictors
X = yelp[features]
y = yelp['good_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg.fit(X_train, y_train)
zip(features, logreg.coef_[0]) # ??? why the [0]

# confusion matrix
y_pred = logreg.predict(X_test)
confusion = metrics.confusion_matrix(y_test, y_pred)
print confusion

# accuracy
print 'accuracy ', metrics.accuracy_score(y_test, y_pred)
#senstivity
print 'sensitivity ', 1.0*confusion[1][1] / (confusion[1][0] + confusion[1][1]) 
#specificity
print 'specificity ', 1.0*confusion[0][0] / (confusion[0][0] + confusion[0][1]) 

# the accuracy only went up a little bit.



##### Part 2 ######

# 1. Read in the titanic data set.
titanic = pd.read_csv('../data/titanic.csv')

# 4. Create a new column called "wife" that is True
# if the name of the person contains Mrs.
# AND their SibSp is at least 1

#each condition
titanic['wife'] = titanic['SibSp']>=1
titanic['wife'] = titanic['Name'].apply(lambda x: x.lower().find('mrs'))>=0

# AND the conditions together
titanic['wife'] = (titanic['Name'].apply(lambda x: x.lower().find('mrs'))>=0) & (titanic['SibSp']>=1)

# 5. What is the average age of a male and
# the average age of a female on board?
males = titanic[titanic.Sex == 'male']
ave_male_age = males.Age.mean() # 30.73

females = titanic[titanic.Sex == 'female']
ave_female_age = females.Age.mean() # 27.92

# 5. Fill in missing MALE age values with the
# average age of the remaining MALE ages

# pseudo code:
# if titanic.Sex == 'male' and titanic.Age == NaN then titanic.Age = ave_male_age

titanic.Age[((titanic.Sex == 'male') & ~(titanic.Age>0))] = ave_male_age

# Patrick's way 
#df['Age'] = df.groupby("Sex").transform(lambda x: x.fillna(x.mean()))['Age']

# 6. Fill in missing FEMALE age values with the
# average age of the remaining FEMALE ages
titanic.Age[((titanic.Sex == 'female') & ~(titanic.Age>0))] = ave_female_age


# 7. Perform a Logistic Regression using
# Survived as your response and age, wife
# as predictors

logreg = LogisticRegression()
features = ['Age', 'wife']
X = titanic[features]
y = titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg.fit(X_train, y_train)
zip(features, logreg.coef_[0])

# 8. Show Accuracy, Sensitivity, Specificity and 
# Confusion matrix
y_pred = logreg.predict(X_test)
confusion = metrics.confusion_matrix(y_test, y_pred)
# accuracy
print 'accuracy ', metrics.accuracy_score(y_test, y_pred)
#senstivity
print 'sensitivity ', 1.0*confusion[1][1] / (confusion[1][0] + confusion[1][1]) 
#specificity
print 'specificity ', 1.0*confusion[0][0] / (confusion[0][0] + confusion[0][1]) 

accuracy = np.absolute(cross_val_score(logreg, X, y, scoring='accuracy', cv=5))
print 'cross-validated accuracy ', accuracy.mean()


# 9. now use ANY of your variables as predictors
# Still using survived as a response to boost metrics!
titanic['is_female'] = (titanic.Sex == 'female')
logreg = LogisticRegression()
features = ['Age', 'wife', 'Pclass', 'Fare', 'is_female']
X = titanic[features]
y = titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg.fit(X_train, y_train)
zip(features, logreg.coef_[0])


# 10. Show Accuracy, Sensitivity, Specificity
y_pred = logreg.predict(X_test)
confusion = metrics.confusion_matrix(y_test, y_pred)
# accuracy
print 'accuracy ', metrics.accuracy_score(y_test, y_pred)
#senstivity
print 'sensitivity ', 1.0*confusion[1][1] / (confusion[1][0] + confusion[1][1]) 
#specificity
print 'specificity ', 1.0*confusion[0][0] / (confusion[0][0] + confusion[0][1]) 
accuracy = np.absolute(cross_val_score(logreg, X, y, scoring='accuracy', cv=5))
print 'cross-validated accuracy ', accuracy.mean()



# REMEMBER TO USE
# TRAIN TEST SPLIT AND CROSS VALIDATION
# FOR ALL METRIC EVALUATION!!!!

