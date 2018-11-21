#!/usr/bin/env python
# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
import time

#x,y = np.loadtxt("test_dat.txt",delimiter=',',unpack=True)
#plt.plot(x,y,'+',color='black')

#plt.show()

def exeTime(func):
    """time consumed"""
    def newFunc(*args,**args2):
        t0 = time.time()
        back = func(*args, **args2)
        return back,time.time() - t0
    return newFunc


def loadDataSet(filename):
    """read data from file"""
    numFeat = len(open(filename).readline().split(','))-1
    x = []
    y = []
    file = open(filename)
    for line in file.readlines():
        lineArr = []
        curLine = line.strip().split(',')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        x.append(float(curLine[0]))
        y.append(float(curLine[-1]))
    return np.mat(x),np.mat(y)


def h(theta,x):  #for one single event
    return (theta.T*x)#[0,0]

def J(theta,x,y):
    n,m = x.shape
    diff = np.array(theta.T*x-y)
    return (diff*diff).sum()/2/m

def standardize(X):
    X = np.array(X)
    n,m = X.shape
    sumX = []
    RMSX = []
    for j in range(n):
        sumX.append(np.mean(X[j,:]))
        RMSX.append(np.std(X[j,:]))
        for k in range(m):
            X[j,k] = (X[j,k]-sumX[j])/RMSX[j]
    return X


def normalization(X):
    X = np.array(X)
    n,m = X.shape
    max = []
    min = []
    for j in range(n):
        max.append(np.max(X[j,:]))
        min.append(np.min(X[j,:]))
        for k in range(m):
            X[j,k]=(X[j,k]-min[j])/(max[j]-min[j])
    return X

@exeTime
def bgd(rate,maxLoop,epsilon,x,y):
    """Batch gradient descent"""
    m,n = x.shape
    theta = np.zeros((m,1))
    count = 0
    converged = False
    error = float('inf')
    errors = []
    thetas = {}
    for j in range(m):
        thetas[j] = [theta[j,0]]
    while count <= maxLoop:
        if(converged):
            break
        count = count + 1
        for j in range(m):  #Loop for every theta
            #print(np.dot(theta.T,x))
            J = np.array(y-np.dot(theta.T,x))
            #print("cost function: ",J)
            deriv = (np.array((y-np.dot(theta.T,x)))*np.array(x[j,:])).sum()/n
            #print("gradient is: ",deriv)
            #print("theta",j)
            #print(theta[j])
            theta[j] = theta[j]+rate*deriv
            #print(theta[j])
            thetas[j].append(theta[j,0])
        error = (J*J).sum()/2/n
        #print("error now is:",error)
        errors.append(error)
        if(error<epsilon):
            converged = True
    return theta,errors,thetas

@exeTime
def sgd(rate,maxLoop,epsilon,x,y):
    m,n = x.shape
    theta = np.zeros((m,1))
    count = 0
    converged = False
    error = float('inf')
    errors = []
    thetas = {}
    for j in range(m):
        thetas[j] = [theta[j,0]]
    while count <= maxLoop:
        if(converged):
            break
        count = count + 1
        errors.append(float('inf'))
        for i in range(n):
            if converged:
                break
            diff = y[0,i]-theta.T*x[:,i]
            for j in range(m):
                theta[j,0] = theta[j,0] + rate*diff*x[j,i]
                thetas[j].append(theta[j,0])
            error = J(theta,x,y)
            errors[-1] = error
        if(error<epsilon):
            converged = True
    return theta,errors,thetas
