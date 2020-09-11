# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:58:17 2020

@author: LAVEENA
"""

import pandas as pd
import matplotlib.pyplot as plt

#importing data
data = pd.read_csv("C://Users/LAVEENA/Desktop/LRdata1.txt")

#slicing data to X and y
X = data.iloc[:,0:2]
y = data.iloc[:,2]

#further slicing into a and b, 2 different features
a = X.iloc[:,0]
b = X.iloc[:,1]

#plotting features on both axis and corresponding labels
fig = plt.figure(figsize=(8,8))
plt.scatter(a, b, c=y)

#splitting data into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#fitting logistic regression into X_train,y_train
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix #prints confusion metrix

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

y_pre = model.predict(X)
plt.scatter(a, b, c=y_pre) #plotting predicted labels



#importing data
data2 = pd.read_csv("C://Users/LAVEENA/Desktop/LRdata2.txt")
X1 = data2.iloc[:,0:2]
y1 = data2.iloc[:,2]


a1 = X1.iloc[:,0]
b1 = X1.iloc[:,1]

plt.scatter(a1, b1, c=y1)
#now we can see that data is not linearly separable,so we add 2 new features which are square of given features
#if we continue with only 2 features, the predicted model will be linear with very low precision and recall
d = a*a
e = b*b

import numpy as np
X2 = np.stack((a1, b1, d, e), axis = 1)
#X2 is new matrix with 4 features (2 were given and 2 made by squaring given 2)


X_train1, X_test1, y_train1, y_test1 = train_test_split(X2, y1, test_size=0.2)

model1 = LogisticRegression(penalty='l2').fit(X_train1,y_train1)
y_pred1 = model1.predict(X_test1)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test1, y_pred1)
cnf_matrix

print("Accuracy:",metrics.accuracy_score(y_test1, y_pred1))
print("Precision:",metrics.precision_score(y_test1, y_pred1))
print("Recall:",metrics.recall_score(y_test1, y_pred1))

y_pred2 = model1.predict(X2)
plt.scatter(a, b, c=y_pred2)










