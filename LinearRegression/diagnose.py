#!/usr/bin/env python
# coding=utf-8

import regression as reg
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import random

X,y = reg.loadDataSet('water.txt')
m,n = X.shape
X = np.concatenate((np.ones((1,n)),X),axis=0)
#print(X)
#print(y)

train_percent = 0.8
trainval_percent = 0.1
test_percent = 0.1

list = list(range(n))

#产生训练集
tr = int(n * train_percent)
X_train = np.array()
train = random.sample(list, tr)
for j in train:
    X_train.append(X[0,j])
    X_train.append(X[1,j])
print(X_train)



#产生交叉验证集
for x in train:
    list.remove(x)
num = len(list)
tv = int(num * trainval_percent/(trainval_percent+test_percent))
trainval = random.sample(list, tv)

#产生测试集
for x in trainval:
    list.remove(x)
test = list
