# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:24:31 2018
@author: Uchiha Madara and Shivam Patel

Title : BlAckfriday  Data Analysis Project (BADA  Project)
"""
#Importing Libraries
import pandas as pd
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
#=========================(Preprocessing)===================================================================
"""
def OHE(df): #One Hot Encoding.
#df = pd.DataFrame(X) only for viewing that X has removed all nan values.
#Convert all categorical data which are either strings or numbers to character labels.
    dummy_variable_1 = pd.get_dummies(df['City_Category'])

    dummy_variable_1.rename(columns = {'A':'City_Category - A',
                                   'B':'City_Category - B', 
                                   'C':'City_Category - C' }, inplace = True)

    df = pd.concat([dummy_variable_1,df], axis = 1)
    df.drop('City_Category', axis = 1, inplace = True)
    dummy_variable_2 = pd.get_dummies(df['Gender'])

    dummy_variable_2.rename(columns = {'M':'Gender - Male',
                                   'F':'Gender - Female'},inplace = True)

    df = pd.concat([dummy_variable_2,df], axis = 1)

    df.drop('Gender', axis = 1, inplace = True)
    dummy_variable_3 = pd.get_dummies(df['Stay_In_Current_City_Years'])

    dummy_variable_3.rename(columns = {'0':'Stay_In_Current_City_Years - 0',
                                   '1':'Stay_In_Current_City_Years - 1',
                                   '2':'Stay_In_Current_City_Years - 2',
                                   '3':'Stay_In_Current_City_Years - 3',
                                   '4+':'Stay_In_Current_City_Years - 4+'}, inplace = True)

    df = pd.concat([dummy_variable_3,df], axis = 1)

    df.drop('Stay_In_Current_City_Years', axis = 1, inplace = True)

# Categorical Variable to Quantitative Variable - Age

    dummy_variable_4 = pd.get_dummies(df['Age'])

    dummy_variable_4.rename(columns = {'0-17':'Age: 0-17',
                                   '18-25':'Age: 18-25',
                                   '26-35':'Age: 26-35',
                                   '36-45':'Age: 36-45',
                                   '46-50':'Age: 46-50',
                                   '51-55':'Age: 51-55',
                                   '55+':'Age: 55+',}, inplace = True)

    df = pd.concat([dummy_variable_4,df], axis = 1)

    df.drop('Age', axis = 1, inplace = True)
    return df
    """
""" Removing the data points with na reduces the df density considerably, this will reduce model performance.
    The data  
    data = df.dropna()
"""
"""
#reading file
df = pd.read_csv('train.csv')

#Preprocessing

#imputing NAs.
imput = Imputer(strategy ='most_frequent' )
df.iloc[: , 9:11] = imput.fit_transform(df.iloc[:,9:11])
df= df.iloc[:,2:]

df = OHE(df)
"""
#===================================(/Preprocessing)=================================================================

# writing preprocessed file : pd.DataFrame.to_csv(df,'Processed_BlackFriday.csv')
df = pd.read_csv('Processed_BlackFriday.csv')
#Data Allocation for regression on Purchase price.

X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

#splitting the data
X_train,X_test, Y_train , Y_test = train_test_split(X,Y,test_size= 0.01, random_state = 0 )

#modelling
#===========================(Regression on Purchase Price:)===========================================================
"""Linear Regression : bad (adj R^2 12%)
from sklearn.linear_model import LinearRegression
lr = LinearRegression( n_jobs = 3)

lr.fit(X_train , Y_train)
Y_pred = lr.predict(X_test)
Y_train_pred = lr.predict(X_train)
"""

"""Polynomial Regression: bad (adj R^2 22%)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)
"""

"""Support Vector Regression : HP Tuning using Grid Search
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
Cs = [0.01, 0.1, 1, 10]
gammas = [0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid,n_jobs = 3,n_folds = 1, verbose = True)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
"""

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 40, max_depth = 20, min_samples_split = 5, n_jobs = 3, verbose = True, random_state = 42)
rfr.fit(X_train,Y_train)
Y_pred = rfr.predict(X_test)
Y_train_pred = rfr.predict(X_train)

#Evaluating test error

print('Evaluating Test Error')
print('MAE: ', metrics.mean_absolute_error(Y_test,Y_pred))
print('MSE: ',metrics.mean_squared_error(Y_test,Y_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))
print('Mean : ', np.mean(Y_test))
print('R^2 Value : ', metrics.r2_score(Y_test,Y_pred))
r_sq = metrics.r2_score(Y_test,Y_pred)
adj_r2 = 1 - (1-r_sq)*(len(Y_test)-1)/(len(Y_test)-X.shape[1]-1)
print('Adj R^2 Value : ', adj_r2)
"""
Evaluating Test Error
MAE:  2176.119725561405
MSE:  8504604.461960774
RMSE:  2916.2654992234116
Mean :  9363.30667151427
R^2 Value :  0.6698514553021874
Adj R^2 Value :  0.6685255575323166

"""
#Evaluating Train Error
print('Evaluating Train Error')
print('MAE: ', metrics.mean_absolute_error(Y_train, Y_train_pred))
print('MSE: ',metrics.mean_squared_error(Y_train,Y_train_pred))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(Y_train, Y_train_pred)))
print('Mean : ', np.mean(Y_train))
print('R^2 Value : ',metrics.r2_score(Y_train,Y_train_pred))
r_sq = metrics.r2_score(Y_train,Y_train_pred)
adj_r2 = 1 - (1-r_sq)*(len(Y_train)-1)/(len(Y_train)-X.shape[1]-1)
print('Adj R^2 Value : ', adj_r2)

"""
Evaluating Train Error
MAE:  1925.798911507789
MSE:  6695165.708176956
RMSE:  2587.501827666399
Mean :  9262.965240273465
R^2 Value :  0.7345894738342462
Adj R^2 Value :  0.7345787510431115
"""
#================================(Classification : Marital status)====================================================
#Data allocation
X = pd.concat([df.iloc[:,0:19] , df.iloc[:,20:]], axis =1)
X = X.iloc[:,:].values
Y = df.iloc[:,19].values

#Splitting
X_train,X_test, Y_train , Y_test = train_test_split(X,Y,test_size= 0.01, random_state = 0 )

#modelling
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', n_jobs = 3, random_state = 42)

rfc.fit(X_train,Y_train)
Y_pred = rfc.predict(X_test)
Y_train_pred = rfc.predict(X_train)

#Evaluating test set.
from sklearn.metrics import  precision_score, accuracy_score, recall_score
print('Confusion Matrix')
print(pd.crosstab(Y_test,Y_pred, margins = True))
print('Accuracy : ',accuracy_score(Y_test,Y_pred)*100, ' %' )
print('Precision : ', precision_score(Y_test,Y_pred, average = 'macro'))
print('Recall : ', recall_score(Y_test, Y_pred, average = 'macro'))

#Evaluating train set.
print('Confusion Matrix')
print(pd.crosstab(Y_train,Y_train_pred, margins = True))
print('Accuracy : ',accuracy_score(Y_train,Y_train_pred)*100, ' %' )
print('Precision : ', precision_score(Y_train,Y_train_pred, average = 'macro'))
print('Recall : ', recall_score(Y_train, Y_train_pred, average = 'macro'))

#Occupation and age is important towards this determination.

















