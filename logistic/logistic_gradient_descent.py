__author__ = 'PC-LiNing'

import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
import numpy as np
from sklearn.cross_validation import train_test_split
import math
import datetime


def Sigmoid(z):
    return float(1.0 / float((1.0 + math.exp(-1.0*z))))


# compute sigmoid(theta*x)
def Hypothesis(theta, x):
    z = 0
    for i in range(len(theta)):
        z += x[i]*theta[i]
    return Sigmoid(z)


# m = len(X)
# loss = - 1/m * sum_loss(X,Y)
def loss(X,Y,theta):
    m = len(Y)
    sumOfErrors = 0
    for i in range(m):
        hi = Hypothesis(theta,X[i])
        error = Y[i] * math.log(hi) if Y[i] == 1 else (1-Y[i]) * math.log(1-hi)
        sumOfErrors += error
    const = -1/m
    return const * sumOfErrors


# compute delta of theta_j
def delta_theta_j(X,Y,theta,j,alpha):
    sum_delta = 0
    m = len(Y)
    for i in range(m):
        hi = Hypothesis(theta,X[i])
        sum_delta += (hi - Y[i])*X[i][j]
    constant = float(alpha)/float(m)
    return constant * sum_delta


# compute delta of theta
def Gradient_Descent(X,Y,theta,alpha):
    m = len(Y)
    new_theta = []
    for j in range(len(theta)):
        CFDerivative = delta_theta_j(X,Y,theta,j,alpha)
        new_theta_value = theta[j] - CFDerivative
        new_theta.append(new_theta_value)
    return new_theta

# evaluation test
def score(test_X,test_Y,theta):
    correct_count = 0
    length = len(test_X)
    for i in range(length):
        prediction = round(Hypothesis(test_X[i],theta))
        if prediction == test_Y[i]:
            correct_count += 1
    acc = float(correct_count) / float(length)
    return acc


# alpha = step length
def Logistic_Regression(X,Y,alpha,theta,num_iters):
    for x in range(num_iters):
        new_theta = Gradient_Descent(X,Y,theta,alpha)
        theta = new_theta
        if x % 100 == 0:
            current_loss = loss(X,Y,theta)
            current_acc = score(X,Y,theta)
            time_str = datetime.datetime.now().isoformat()
            print("{}: iter {}, loss {:g}, acc {:g}".format(time_str, x, current_loss, current_acc))
    return theta

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

df = pd.read_csv("data.csv",header=0)
df.columns = ['grade1','grade2','label']
X = df[["grade1","grade2"]]
X = np.array(X)
# X = [100,2]
X = min_max_scaler.fit_transform(X)

Y = df["label"].map(lambda x: float(x.rstrip(';')))
# Y = [100,]
Y = np.array(Y)

# creating testing and training set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)
initial_theta = [0,0]
alpha = 0.1
iterations = 1000
result_theta = Logistic_Regression(X_train,Y_train,alpha,initial_theta,iterations)
result_acc = score(X_test,Y_test,result_theta)
print("Acc of test: "+str(result_acc))