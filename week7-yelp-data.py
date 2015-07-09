# -*- coding: utf-8 -*-
"""
Week 7 optional homework 
Yelp data set
yelp.csv contains the Yelp ratings data

Each observation in this dataset is a review of a particular business by a 
 particular user.
The "stars" column is the number of stars (1 through 5) assigned by the reviewer 
 to the business. (Higher stars is better.)
The "cool" column is the number of "cool" votes this particular review received
 from other Yelp users. There is no limit to how many "cool" votes a review can
 receive.
The "useful" and "funny" columns are similar to the "cool" column.

Homework tasks:
Read yelp.csv into a DataFrame.
Explore the relationship between each of the vote types (cool/useful/funny) and the number of stars.
Define cool/useful/funny as the features, and stars as the response.
Fit a linear regression model and interpret the coefficients. Do the coefficients make intuitive sense to you? Explore the Yelp website to see if you detect similar trends.
Evaluate the model by splitting it into training and testing sets and computing the RMSE. Does the RMSE make intuitive sense to you?
Try removing some of the features and see if the RMSE improves.
Bonus: Think of some new features you could create from the existing data that might be predictive of the response. (This is called "feature engineering".) Figure out how to create those features in Pandas, add them to your model, and see if the RMSE improves.
Bonus: Compare your best RMSE on testing set with the RMSE for the "null model", which is the model that ignores all features and simply predicts the mean rating in the training set for all observations in the testing set.
Bonus: Instead of treating this as a regression problem, treat it as a classification problem and see what testing accuracy you can achieve with KNN.
Bonus: Figure out how to use linear regression for classification, and compare its classification accuracy to KNN.

"""
# Read yelp.csv into a DataFrame.


# Explore the relationship between each of the vote types (cool/useful/funny) and the number of stars.
# Define cool/useful/funny as the features, and stars as the response.
# Fit a linear regression model and interpret the coefficients. Do the coefficients make intuitive sense to you? Explore the Yelp website to see if you detect similar trends.
# Evaluate the model by splitting it into training and testing sets and computing the RMSE. Does the RMSE make intuitive sense to you?
# Try removing some of the features and see if the RMSE improves.
# Bonus: Think of some new features you could create from the existing data that might be predictive of the response. (This is called "feature engineering".) Figure out how to create those features in Pandas, add them to your model, and see if the RMSE improves.
# Bonus: Compare your best RMSE on testing set with the RMSE for the "null model", which is the model that ignores all features and simply predicts the mean rating in the training set for all observations in the testing set.
# Bonus: Instead of treating this as a regression problem, treat it as a classification problem and see what testing accuracy you can achieve with KNN.
# Bonus: Figure out how to use linear regression for classification, and compare its classification accuracy to KNN.

