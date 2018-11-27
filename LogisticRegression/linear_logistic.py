#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import time


def exeTime(func):
    """time consumed"""
    def newFunc(*args,**args2):
        t0 = time.time()
        back = func(*args, **args2)
        return back,time.time() - t0
    return newFunc


def loadDataSet(filename):
    X = []
    y = []
    num=len(open(filename).readline().split(','))-1
    file = open(filename)
    #print("File opened")
    for line in file.readlines():
        lineArr = []
        curLine = line.strip().split(',')
        for i in range(num+1):
            lineArr.append(float(curLine[i]))
        for j in range(num):
            X.append(lineArr[j])
        y.append(lineArr[-1])
    m = len(y)
    d = np.array(X).reshape(m,2)
    #print(d.shape)
    X = np.concatenate((np.ones((m,1)),d),axis=1)
    return X,np.array(y)


def h(x,theta):
    """sigmoid function"""
    z = np.dot(x.T,theta)
    return 1/(1+np.exp(-z))

def sigmoid1(z):
    return 1.0/(1.0+np.exp(-z))



@exeTime
def logistic_bgd(rate,maxLoop,epsilon,X,y):
    n = X.shape #n = (100,2) for ex2data1.txt
    theta = np.zeros((n[-1],1))
    #print(X)
    #print(y)
    y = y.reshape(n[0],1)
    #print(y[1])
    count = 0
    converged = False
    error = float('inf')
    errors = []
    thetas = {}
    for j in range(n[-1]):
        thetas[j] = [theta[j,0]]
    while count <= maxLoop:
        if(converged):
            break
        count = count + 1
        #print(X)
        #print("theta now is ",theta)
        sigmoid = np.array(h(X.T,theta))
        #print("sigmoid for now is ",sigmoid)
        #print("gradient is ",np.dot(X.T,(y-sigmoid)))
        J = -(1/n[0])*((np.log(sigmoid)*y+(1-y)*np.log(1-sigmoid))).sum()
        #print("cost function now is ",J)
    #print("X shape is ",X.shape)
    #print("diff shape is: ",(y-sigmoid).shape)
        theta = theta + rate * np.dot(X.T,(y-sigmoid))
        for k in range(n[-1]):
            thetas[k].append(theta[k,0])
        error = J
        errors.append(error)
        #print(theta)
    return theta,errors,thetas

@exeTime
def logistic_sgd(rate,maxLoop,epsilon,X,y):
    n = X.shape #n = (100,2) for ex2data1.txt
    theta = np.ones((n[-1],1))
    #print(n[0])
    y = y.reshape(n[0],1)
    #print(y[1])
    count = 0
    converged = False
    error = float('inf')
    errors = []
    thetas = {}
    for j in range(n[-1]):
        thetas[j] = [theta[j,0]]
    while count <= maxLoop:
        #print(count)
        if(converged):
            break
        for j in range(n[0]):
            #print('Now we use event # ',j)
            count = count + 1
            sigmoid = h(X[j].T,theta)
            J = -np.log(sigmoid[0])*y[j]-(1-y[j])*np.log(1-sigmoid[0])
            #print((rate*sigmoid[0]*X[j]).T)
            for k in range(n[-1]):
                theta[k] = theta[k] + rate * (y[j]-sigmoid[0]) * X[j,k]
                thetas[k].append(theta[k,0])
            #print(theta)
            error = J
            errors.append(error[0])
            #thetas.append(theta)
            #if error<epsilon:
            #    break
    return theta,errors,thetas


def oneVsAll(rate,maxLoop,epsilon,X,y):
    # 类型数
    classes = set(np.ravel(y))
    # 决策边界矩阵
    Thetas = np.zeros((len(classes)),X.shape[-1])
    for idx, c in enumerate(classes):
        newY = np.zeros(y.shape)
        newY[np.where(y == c)] = 1
        result,timeConsumed = bgd(rate,maxLoop,epsilon,X,y)
        theta,errors,thetas = results
        Thetas[idx] = thetas[-1].ravel()
    return Thetas


def predictOneVsAll(X,Thetas):
    H = sigmoid1(Thetas*X.T)
    return H 






"""
X,y = loadDataSet("linear_sample.txt")
rate = 0.01
maxLoop = 10000
epsilon = 0.01


results,timeConsumed = logistic_sgd(rate,maxLoop,epsilon,X,y)
theta,errors,thetas= results
print(thetas[0])


score1_fal = []
score2_fal = []
score1_pass = []
score2_pass = []

score1,score2,result = np.loadtxt("linear_sample.txt",delimiter=',',unpack=True)
for i in range(len(result)):
    if result[i] == 1:
        score1_pass.append(score1[i])
        score2_pass.append(score2[i])
    else:
        score1_fal.append(score1[i])
        score2_fal.append(score2[i])

plt.plot(score1_pass,score2_pass,'+',color='blue')
plt.plot(score1_fal,score2_fal,'*',color='red')
xx = np.linspace(0,1,100)
theta1 = -(theta[0]+theta[1]*xx)/theta[2]
plt.plot(xx,theta1,color='black')
plt.show()


errorFig = plt.figure()
title = 'error decrease with iterations'
ax = errorFig.add_subplot(111,title=title)
ax.plot(range(len(thetas[2])), thetas[2])
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Cost J')
plt.show()
"""























"""
@exeTime
def gradient(X,y,options):

    n = X.shape #n = (# of samples,# of features)
    theta = np.ones((n[-1],1))
    count = 1
    error = float('inf')
    errors = []
    thetas = []
    rate = options.get('rate',0.01)
    epsilon = options.get('epsilon',0.1)
    maxLoop = options.get('maxLoop',1000)
    theLambda = options.get('theLambda',0)
    method = options['method']

    def _bgd(theta):
        converged = False
        for i in range(maxLoop):
            if converged:
                break
            count = count + 1
            sigmoid = np.array(h(X.T,theta))
            J = (1/n[0])*((np.log(sigmoid)*y+(1-y)*np.log(1-sigmoid))).sum()
            theta = theta + rate * np.dot(X.T,(y-sigmoid))
            error = J
            errors.append(error)
            thetas.append(theta)
            if error < epsilon:
                break
        return thetas,errors,i+1

    def _sgd(theta):
        for i in range(maxLoop):
            if converged:
                break
            count = count + 1
            for j in range(n[0]:
                sigmoid = h(x[j].T,theta)

"""
