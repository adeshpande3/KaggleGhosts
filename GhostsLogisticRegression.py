from __future__ import division
from pandas import DataFrame, Series
from sklearn.cross_validation import train_test_split
import scipy
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn import linear_model
import csv
import sys

# This program will use Logistic Regression as a machine learning algorithm that predicts
# whether a set of input characteristics represents a ghost, ghoul, or goblin

Xtrain=[] # Will contain the characteristic data for each monster. Xtrain will 
		  # contain info about the monster's bone length, rotting flesh, 
		  # hair length, soul, and color. 
Ytrain=[] # WIll contain a binary label for what type of monster it is 
Xtest=[] # Will contain the characteristic data for the monsters in the test set

Xtrain = pd.read_csv("train.csv")
Ytrain = Xtrain['type']
Xtrain['type'].replace(['Ghoul','Goblin','Ghost'],[0,1,2],inplace=True)
Xtrain.pop('type')
Xtrain.pop('id')
Xtrain = pd.get_dummies(Xtrain,columns=['color'])
Ytrain = pd.np.array(Ytrain)
Xtrain = pd.np.array(Xtrain)

##################################################################################
# If you want to run some cross validation with the training data

#local_train, local_test = train_test_split(Xtrain,test_size=0.1,random_state=123)
#local_train_y = local_train['type']
#local_train_x = local_train.drop(['type'],axis=1)
#local_test_y = local_test['type']
#local_test_x = local_test.drop('type',axis=1)

#logistic = linear_model.LogisticRegression()
#logistic.fit(local_train_x, local_train_y)
#preds = logistic.predict(local_test_x)
#print (np.mean((preds) == local_test_y))

##################################################################################

Xtest = pd.read_csv("test.csv")
idList = Xtest['id']
Xtest.pop('id')
Xtest = pd.get_dummies(Xtest,columns=['color'])
Xtest = pd.np.array(Xtest)

#model = sm.OLS(Ytrain,Xtrain)
#result = model.fit()
#preds = result.predict(Xtest)

logistic = linear_model.LogisticRegression()
logistic.fit(Xtrain, Ytrain)
preds = logistic.predict(Xtest)
df1 = pd.DataFrame({'labels': preds})
df1['labels'].replace([0,1,2],['Ghoul','Goblin','Ghost'],inplace=True)
preds = df1['labels'].tolist()	
results = [[0 for x in range(2)] for x in range(len(preds))]
for index in range(0,len(preds)):
	results[index][0] = idList[index]
	results[index][1] = int(round(preds[index]))

results = pd.np.array(results)
with open("result.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(results)
