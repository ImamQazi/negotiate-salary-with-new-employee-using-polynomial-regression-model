# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 20:50:00 2021

@author: Imam Qazi
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data set
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values #metrix
y = dataset.iloc[:, 2].values   #vector


# Fiting the Linear Regression dataset
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)


# Fiting the Polynomial Regression dataset
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 2)
x_pr = pr.fit_transform(x)

#now create second linear regression object to fit x_pr
lr_pr = LinearRegression()
lr_pr.fit(x_pr,y)

#Visualising the linear Regression results
plt.scatter(x,y, color='red')
plt.plot(x, lr.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Poition Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression results
plt.scatter(x,y, color='red')
plt.plot(x, lr_pr.predict(pr.fit_transform(x)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Poition Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
pred = lr.predict(np.array([6.5]).reshape(1,1))

#Predicting a new result with Polynomial Regression
pred_poly = lr_pr.predict(pr.fit_transform(np.array([6.5]).reshape(1,1)))