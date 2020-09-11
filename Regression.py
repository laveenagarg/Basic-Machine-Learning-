# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:11:48 2020

@author: LAVEENA
"""

#I had data saved on my desktop, you can change it accordingly
import pandas as pd
data = pd.read_csv("C://Users/LAVEENA/Desktop/Reg_data1.txt")


#creating X and y vectors
X = data.iloc[:,0]
y = data.iloc[:,1]

#looking at data by plotting graphs
data.plot(kind = 'line')
data.plot(kind = 'bar')
data.plot(kind = 'box')
data.plot(kind = 'barh')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Using Linear regression inbuild model from scipy
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(X_train, y_train)

#plotting original data and fitted line
import matplotlib.pyplot as plt
plt.plot(X, y, 'o', label='original data')
plt.plot(X, intercept + slope*X, 'r', label='fitted line')
plt.legend()
plt.show()




